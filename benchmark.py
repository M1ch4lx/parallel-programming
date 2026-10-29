from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return values


def run_once(
    launcher: str,
    launcher_args: list[str],
    processes: int,
    n: int,
    a: float,
    p: float,
    tension: float,
    tol: float,
    max_iters: int,
    solver: Path,
) -> dict:
    cmd = [
        launcher,
        *launcher_args,
        "-n",
        str(processes),
        sys.executable,
        str(solver),
        "--n",
        str(n),
        "--a",
        str(a),
        "--p",
        str(p),
        "--tension",
        str(tension),
        "--tol",
        str(tol),
        "--max-iters",
        str(max_iters),
        "--json",
    ]

    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        "Could not find JSON output from solver.\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run strong-scaling benchmark for the membrane MPI solver.")
    parser.add_argument("--sizes", type=parse_int_list, default=parse_int_list("257,513"), help="Problem sizes, e.g. 257,513")
    parser.add_argument("--processes", type=parse_int_list, default=parse_int_list("1,2,4,8"), help="MPI process counts, e.g. 1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per (N,P); median time is stored.")
    parser.add_argument("--launcher", default="mpiexec", help="MPI launcher executable (default: mpiexec).")
    parser.add_argument("--launcher-arg", action="append", default=[], help="Extra launcher argument; repeat this option as needed.")
    parser.add_argument("--solver", type=Path, default=here / "membrane_mpi.py")
    parser.add_argument("--output", type=Path, default=here / "benchmark.csv")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--tension", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--max-iters", type=int, default=200_000)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for n in args.sizes:
        for proc_count in args.processes:
            results = []
            for rep in range(1, args.repeats + 1):
                result = run_once(
                    args.launcher,
                    args.launcher_arg,
                    proc_count,
                    n,
                    args.a,
                    args.p,
                    args.tension,
                    args.tol,
                    args.max_iters,
                    args.solver,
                )
                results.append(result)
                print(
                    f"N={n:5d}, P={proc_count:3d}, repeat={rep}/{args.repeats}: "
                    f"{result['time_seconds']:.6f} s"
                )

            times = [float(r["time_seconds"]) for r in results]
            representative = min(results, key=lambda r: abs(float(r["time_seconds"]) - statistics.median(times)))
            rows.append(
                {
                    "n": n,
                    "processes": proc_count,
                    "time_seconds": statistics.median(times),
                    "time_min_seconds": min(times),
                    "time_max_seconds": max(times),
                    "iterations": representative["iterations"],
                    "max_update": representative["max_update"],
                    "process_grid": "x".join(map(str, representative["process_grid"])),
                }
            )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark data to: {args.output}")


if __name__ == "__main__":
    main()
