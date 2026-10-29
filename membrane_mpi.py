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