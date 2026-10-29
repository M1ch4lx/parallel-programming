from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI

from membrane_core import jacobi_serial, save_visualizations, validate_problem


def split_range(length: int, parts: int, coord: int) -> tuple[int, int]:
    """Balanced [start, stop) partition of range(length)."""
    base, remainder = divmod(length, parts)
    start = coord * base + min(coord, remainder)
    stop = start + base + (1 if coord < remainder else 0)
    return start, stop


def allocate_shared_matrix(shared_comm: MPI.Comm, n: int) -> tuple[np.ndarray, MPI.Win]:
    """Allocate one n x n float64 array shared by all ranks on one node."""
    rank = shared_comm.Get_rank()
    itemsize = np.dtype(np.float64).itemsize
    nbytes = n * n * itemsize if rank == 0 else 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=shared_comm)
    buffer, disp_unit = win.Shared_query(0)
    if disp_unit != itemsize:
        raise RuntimeError(f"Unexpected shared-window disp_unit={disp_unit}, expected {itemsize}.")

    array = np.ndarray(buffer=buffer, dtype=np.float64, shape=(n, n))
    return array, win


def solve_shared_jacobi(
    n: int,
    a: float,
    p: float,
    tension: float,
    tol: float,
    max_iters: int,
    comm: MPI.Comm,
) -> tuple[np.ndarray | None, int, float, float, tuple[int, int]]:
    """Parallel Jacobi solver using MPI-3 shared-memory windows.

    Important: all ranks must run on the same physical shared-memory node.
    The global grid is stored twice (old/new) and each rank updates only its
    own 2D block. The block grid is chosen with MPI.Compute_dims so it is as
    square as possible for the given number of processes.
    """
    validate_problem(n, a, p, tension)
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Preserve world-rank order inside the node-local communicator.
    shared_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    if shared_comm.Get_size() != size:
        if rank == 0:
            print(
                "ERROR: shared-table mode requires all MPI ranks on one physical node. "
                "Run the job with one node (e.g. #SBATCH --nodes=1).",
                file=sys.stderr,
            )
        comm.Abort(2)

    dims = tuple(MPI.Compute_dims(size, [0, 0]))
    cart = comm.Create_cart(dims=dims, periods=[False, False], reorder=False)
    coords = cart.Get_coords(rank)

    row_start, row_stop = split_range(n, dims[0], coords[0])
    col_start, col_stop = split_range(n, dims[1], coords[1])

    # We never update the fixed boundary h=0.
    i0 = max(row_start, 1)
    i1 = min(row_stop, n - 1)
    j0 = max(col_start, 1)
    j1 = min(col_stop, n - 1)

    old, win_old = allocate_shared_matrix(shared_comm, n)
    new, win_new = allocate_shared_matrix(shared_comm, n)

    # MPICH/Hydra requires MPI_Win_sync to be executed inside a valid
    # passive-target RMA epoch. Keep both shared windows locked for the
    # whole Jacobi solve; Lock_all is called once, not once per iteration.
    win_old.Lock_all()
    win_new.Lock_all()

    if shared_comm.Get_rank() == 0:
        old.fill(0.0)
        new.fill(0.0)

    # Portable load/store visibility for shared MPI windows:
    # writer Win.Sync -> process synchronization -> reader Win.Sync.
    win_old.Sync()
    win_new.Sync()
    shared_comm.Barrier()
    win_old.Sync()
    win_new.Sync()

    dx = a / (n - 1)
    rhs_term = dx * dx * p / tension

    comm.Barrier()
    t0 = MPI.Wtime()
    global_update = float("inf")

    for iteration in range(1, max_iters + 1):
        if i0 < i1 and j0 < j1:
            new[i0:i1, j0:j1] = 0.25 * (
                old[i0 - 1 : i1 - 1, j0:j1]
                + old[i0 + 1 : i1 + 1, j0:j1]
                + old[i0:i1, j0 - 1 : j1 - 1]
                + old[i0:i1, j0 + 1 : j1 + 1]
                + rhs_term
            )
            local_update = float(
                np.max(np.abs(new[i0:i1, j0:j1] - old[i0:i1, j0:j1]))
            )
        else:
            local_update = 0.0

        # Make each rank's stores visible, synchronize all ranks, then refresh
        # the reader view before the next Jacobi iteration.
        win_new.Sync()
        shared_comm.Barrier()
        win_new.Sync()

        global_update = comm.allreduce(local_update, op=MPI.MAX)

        old, new = new, old
        win_old, win_new = win_new, win_old

        if global_update < tol:
            break

    elapsed_local = MPI.Wtime() - t0
    elapsed = comm.allreduce(elapsed_local, op=MPI.MAX)

    # Ensure rank 0 sees the final stores before copying/saving.
    win_old.Sync()
    shared_comm.Barrier()
    win_old.Sync()

    field = old.copy() if rank == 0 else None

    # Close the passive-target epochs before freeing the windows.
    win_old.Unlock_all()
    win_new.Unlock_all()
    win_old.Free()
    win_new.Free()
    cart.Free()
    shared_comm.Free()

    return field, iteration, global_update, elapsed, dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel Poisson solver for a uniformly loaded square membrane."
    )
    parser.add_argument("--n", type=int, default=129, help="Grid size n x n (default: 129).")
    parser.add_argument("--a", type=float, default=1.0, help="Membrane side length a.")
    parser.add_argument("--p", type=float, default=1.0, help="Uniform pressure p.")
    parser.add_argument("--tension", "--T", dest="tension", type=float, default=1.0, help="Membrane tension T.")
    parser.add_argument("--tol", type=float, default=1.0e-8, help="Jacobi stopping tolerance (max update).")
    parser.add_argument("--max-iters", type=int, default=200_000)
    parser.add_argument("--verify", action="store_true", help="On rank 0, compare with serial Jacobi result.")
    parser.add_argument("--plot-prefix", type=str, default=None, help="Save heatmap and 3D plot with this file prefix.")
    parser.add_argument("--save-npy", type=str, default=None, help="Save final grid as .npy on rank 0.")
    parser.add_argument("--json", action="store_true", help="Print one machine-readable JSON result line on rank 0.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    field, iterations, max_update, elapsed, dims = solve_shared_jacobi(
        n=args.n,
        a=args.a,
        p=args.p,
        tension=args.tension,
        tol=args.tol,
        max_iters=args.max_iters,
        comm=comm,
    )

    verification_error = None
    serial_time = None
    if rank == 0:
        if args.verify:
            serial = jacobi_serial(
                args.n, args.a, args.p, args.tension, args.tol, args.max_iters
            )
            verification_error = float(np.max(np.abs(field - serial.field)))
            serial_time = serial.elapsed

        if args.save_npy:
            out = Path(args.save_npy)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, field)

        if args.plot_prefix:
            save_visualizations(field, args.a, args.plot_prefix)

        result = {
            "n": args.n,
            "processes": comm.Get_size(),
            "process_grid": list(dims),
            "iterations": iterations,
            "max_update": max_update,
            "time_seconds": elapsed,
            "max_h": float(np.max(field)),
            "center_h": float(field[args.n // 2, args.n // 2]),
        }
        if verification_error is not None:
            result["verification_max_abs_error"] = verification_error
            result["serial_time_seconds"] = serial_time

        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print("=== Square membrane / MPI shared-memory Jacobi ===")
            print(f"grid                 : {args.n} x {args.n}")
            print(f"MPI processes        : {comm.Get_size()}")
            print(f"2D process grid      : {dims[0]} x {dims[1]}")
            print(f"iterations           : {iterations}")
            print(f"max update           : {max_update:.6e}")
            print(f"parallel time [s]    : {elapsed:.6f}")
            print(f"max h                : {np.max(field):.9e}")
            print(f"center h             : {field[args.n // 2, args.n // 2]:.9e}")
            if verification_error is not None:
                print(f"serial comparison err: {verification_error:.6e}")
                print(f"serial time [s]      : {serial_time:.6f}")


if __name__ == "__main__":
    main()
