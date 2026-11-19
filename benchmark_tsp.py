#!/usr/bin/env python3
"""Pomiary calkowitego czasu MPI (wraz ze startem procesow MPI) dla TSP B&B.

Zapisuje pomiary do CSV i JSON; po pomiarach generuje 6 wykresow
oraz raport za pomoca plot_tsp.py.
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIELDS = [
    'repeat', 'n', 'seed', 'radius', 'bound', 'depth', 'workers', 'mpi_processes',
    'task_count', 'wall_s', 'internal_s', 'prefix_generation_s',
    'nodes', 'pruned', 'leaves', 'best_cost',
]


def integer_list(s):
    result = sorted({int(x.strip()) for x in s.split(',')})
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError('Oczekiwana lista liczb calkowitych >0, np. 1,2,4,8')
    return result


def parse_args():
    ap = argparse.ArgumentParser(description='Dwa eksperymenty TSP B&B na MPI + 6 wykresow')
    ap.add_argument('--n', type=int, default=13)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--radius', type=float, default=100.0)
    ap.add_argument('--bound', choices=['minedge', 'degree'], default='minedge')
    ap.add_argument('--depth-fixed', type=int, default=3)
    ap.add_argument('--workers', type=integer_list, default=[1, 2, 4, 8])
    ap.add_argument('--workers-fixed', type=int, default=8)
    ap.add_argument('--depths', type=integer_list, default=[2, 3, 4])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--outdir', default='wyniki_tsp')
    ap.add_argument('--mpiexec', default='mpiexec')
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error('--repeats musi byc >= 1')
    if 1 not in args.workers:
        ap.error('--workers musi zawierac 1: potrzebujemy T1(d)')
    if not (1 <= args.depth_fixed <= args.n) or any(d > args.n for d in args.depths):
        ap.error('Glebokosci d musza byc mniejsze lub rowne N')
    allocated = os.environ.get('SLURM_NTASKS')
    max_p = max(max(args.workers), args.workers_fixed)
    if allocated and max_p + 1 > int(allocated):
        ap.error(f'Potrzeba >= {max_p+1} procesow MPI, SLURM_NTASKS={allocated}')
    return args


def main():
    a = parse_args()
    folder = Path(a.outdir)
    (folder / 'logs').mkdir(parents=True, exist_ok=True)
    (folder / 'runs').mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('tsp_bb.py').resolve()
    plot_script = Path(__file__).with_name('plot_tsp.py').resolve()
    if not script.exists():
        raise FileNotFoundError(script)

    # Eksperyment 1: stale d, p=1,2,4,8.
    # Eksperyment 2: stale p, zmienne d; DLA KAZDEGO d mierzymy tez p=1.
    cases = {(a.depth_fixed, p) for p in a.workers}
    cases |= {(d, p) for d in a.depths for p in (1, a.workers_fixed)}
    cases = sorted(cases)
    print('Plany (d, liczba workerow):', cases, flush=True)
    print('Czas T liczony zewnetrznie: obejmuje mpiexec + generowanie prefiksow + MPI.', flush=True)

    raw_path = folder / 'measurements.csv'
    with raw_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        expected_opt = None
        for repeat in range(1, a.repeats + 1):
            for depth, workers in cases:
                name = f'n{a.n}_d{depth}_p{workers}_r{repeat}'
                result_path = (folder / 'runs' / f'{name}.json').resolve()
                log_path = folder / 'logs' / f'{name}.log'
                cmd = [
                    a.mpiexec, '-n', str(workers + 1),
                    sys.executable, '-m', 'mpi4py.futures', str(script),
                    '--mode', 'mpi', '--n', str(a.n), '--seed', str(a.seed),
                    '--radius', str(a.radius), '--bound', a.bound,
                    '--depth', str(depth), '--output', str(result_path),
                ]
                print(f'[powt. {repeat}/{a.repeats}] d={depth} p={workers} START', flush=True)
                env = os.environ.copy()
                env['OMP_NUM_THREADS'] = '1'
                env['OPENBLAS_NUM_THREADS'] = '1'
                env['MKL_NUM_THREADS'] = '1'
                start = time.perf_counter()
                with log_path.open('w', encoding='utf-8') as log:
                    log.write('KOMENDA: ' + ' '.join(cmd) + '\n')
                    log.flush()
                    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                wall_s = time.perf_counter() - start
                if proc.returncode:
                    raise RuntimeError(f'MPI zakonczyl sie kodem {proc.returncode}: {log_path}')
                with result_path.open(encoding='utf-8') as f:
                    payload = json.load(f)
                if not payload['verified'] or payload['workers'] != workers:
                    raise RuntimeError(f'Bledna weryfikacja wyniku lub liczba workerow: {result_path}')
                if expected_opt is None:
                    expected_opt = payload['best_cost']
                if not math.isclose(payload['best_cost'], expected_opt, rel_tol=1e-9, abs_tol=1e-7):
                    raise RuntimeError(f'Rozne wyniki optymalne dla tego samego zadania: {result_path}')
                row = {
                    'repeat': repeat, 'n': a.n, 'seed': a.seed,
                    'radius': a.radius, 'bound': a.bound,
                    'depth': depth, 'workers': workers, 'mpi_processes': workers + 1,
                    'task_count': payload['task_count'],
                    'wall_s': f'{wall_s:.9f}',
                    'internal_s': f'{payload["wall_internal_s"]:.9f}',
                    'prefix_generation_s': f'{payload["prefix_generation_s"]:.9f}',
                    'nodes': payload['nodes'], 'pruned': payload['pruned'],
                    'leaves': payload['leaves'], 'best_cost': f'{payload["best_cost"]:.12f}',
                }
                writer.writerow(row)
                output.flush()  # zapis na biezaco, przydatny przy przerwaniu joba
                print(f'  -> T={wall_s:.3f}s, koszt={payload["best_cost"]:.6f}, '
                      f'wezly={payload["nodes"]}, taski={payload["task_count"]}', flush=True)

    subprocess.run([
        sys.executable, str(plot_script), '--input', str(raw_path),
        '--outdir', str(folder), '--depth-fixed', str(a.depth_fixed),
        '--workers', ','.join(map(str, a.workers)),
        '--workers-fixed', str(a.workers_fixed),
        '--depths', ','.join(map(str, a.depths)),
    ], check=True)
    print(f'GOTOWE: {raw_path} oraz 6 wykresow PNG w {folder}/', flush=True)


if __name__ == '__main__':
    main()
