#!/usr/bin/env python3
"""Rownolegly Branch and Bound dla TSP (miasta na okregu).

Uruchomienie na MPI:
    mpiexec -n 9 python -m mpi4py.futures tsp_bb.py --n 13 --depth 3

MPI: jeden master (rank 0) + osiem workerow. Z rownoleglej funkcji
solve_subtree() korzysta MPIPoolExecutor.map(..., chunksize=1).
Pruning i najlepsze rozwiazanie sa WYLACZNIE lokalne dla danego zadania.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

# Dane dostarczane workerom przez initializer (nie przesylamy macierzy
# odleglosci na kazdym tasku osobno).
_DIST = None
_BOUND = 'minedge'


def make_problem(n: int, seed: int, radius: float):
    """Losuje n roznych punktow na okregu i wyznacza macierz odleglosci."""
    rng = random.Random(seed)
    angles = [rng.random() * 2 * math.pi for _ in range(n)]
    points = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    distances = tuple(
        tuple(math.hypot(x1 - x2, y1 - y2) for x2, y2 in points)
        for x1, y1 in points
    )
    return points, angles, distances


def generate_prefixes(n: int, depth: int):
    """Miasta w kodzie to 0..n-1, a w sprawozdaniu 1..n.

    n=4, depth=2 -> (0,1), (0,2), (0,3), czyli [1,2],[1,3],[1,4].
    depth oznacza liczbe miast w prefiksie, lacznie z miastem nr 1.
    """
    return [(0,) + tail for tail in itertools.permutations(range(1, n), depth - 1)]


def init_worker(distances, bound):
    global _DIST, _BOUND
    _DIST = distances
    _BOUND = bound


def _rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0  # lokalny test w trybie --mode serial nie wymaga mpi4py


def solve_subtree(prefix: tuple[int, ...]):
    """Pelne B&B w poddrzewie, z LOKALNYM incumbent (brak komunikacji).

    Zwraca wynik oraz liczniki do analizy obciazenia procesow.
    """
    started = time.perf_counter()
    dist = _DIST
    if dist is None:
        raise RuntimeError('Nie zainicjowano workera: init_worker().')
    n = len(dist)
    all_visited = (1 << n) - 1
    path = list(prefix)
    visited = 0
    partial_length = 0.0
    for city in path:
        visited |= 1 << city
    for a, b in zip(path, path[1:]):
        partial_length += dist[a][b]

    # Gorna granica pochodzi z legalnej, zachlannej kompletacji TEGO prefiksu.
    # NIGDY nie jest wymieniana pomiedzy taskami/workermi.
    greedy_path = list(prefix)
    greedy_seen = visited
    greedy_cost = partial_length
    while greedy_seen != all_visited:
        current = greedy_path[-1]
        next_city = min(
            (v for v in range(1, n) if not (greedy_seen & (1 << v))),
            key=lambda v: (dist[current][v], v),
        )
        greedy_cost += dist[current][next_city]
        greedy_seen |= 1 << next_city
        greedy_path.append(next_city)
    greedy_cost += dist[greedy_path[-1]][0]
    best_cost = greedy_cost
    best_path = tuple(greedy_path) + (0,)

    # Dla porownania: dwie poprawne (dopuszczalne) dolne granice.
    # minedge: dla kazdego nieodwiedzonego miasta co najmniej jedna
    # wychodzaca krawedz do innego nieodwiedzonego lub do startu;
    # dodatkowo co najmniej jedna krawedz od konca aktualnej sciezki.
    # degree: polowa sumy minimalnych brakujacych stopni w otwartym cyklu.
    def lower_bound(last, remaining, cost):
        if not remaining:
            return cost + dist[last][0]
        tail = [i for i in range(1, n) if remaining & (1 << i)]
        if _BOUND == 'minedge':
            result = cost + min(dist[last][v] for v in tail)
            for u in tail:
                # U musi wyjsc do innego nieodwiedzonego lub do miasta 1.
                result += min([dist[u][0]] + [dist[u][v] for v in tail if v != u])
            return result
        # Dolna granica ze stopni w grafie dopelniajacym sciezke:
        # start i last potrzebuja po 1 krawedzi, kazdy inny wierzcholek 2.
        result = min(dist[0][v] for v in tail) + min(dist[last][v] for v in tail)
        for u in tail:
            options = sorted(
                [dist[u][0], dist[u][last]]
                + [dist[u][v] for v in tail if v != u]
            )
            result += options[0] + options[1]
        return cost + 0.5 * result

    nodes = 0
    pruned = 0
    leaves = 0
    improved = 0

    def dfs(last, seen, cost):
        nonlocal nodes, pruned, leaves, best_cost, best_path, improved
        nodes += 1
        remaining = all_visited & ~seen
        if not remaining:
            leaves += 1
            candidate = cost + dist[last][0]
            if candidate < best_cost - 1e-10:
                best_cost = candidate
                best_path = tuple(path) + (0,)
                improved += 1
            return
        if lower_bound(last, remaining, cost) >= best_cost - 1e-10:
            pruned += 1
            return

        # Kolejnosc DFS poprawia znalezienie lokalnego dobrego rozwiazania.
        next_cities = [i for i in range(1, n) if remaining & (1 << i)]
        next_cities.sort(key=lambda i: (dist[last][i], i))
        for city in next_cities:
            path.append(city)
            dfs(city, seen | (1 << city), cost + dist[last][city])
            path.pop()

    dfs(path[-1], visited, partial_length)
    return {
        'prefix': [v + 1 for v in prefix],
        'best_cost': best_cost,
        'best_tour': [v + 1 for v in best_path],
        'rank': _rank(),
        'nodes': nodes,
        'pruned': pruned,
        'leaves': leaves,
        'improvements': improved,
        'task_s': time.perf_counter() - started,
    }


def optimum_circle_length(angles, distances):
    """Niezalezna kontrola wyniku: obwod wielokata wypuklego.

    Ta wartosc NIE jest uzywana przez branch-and-bound ani pruning.
    """
    order = sorted(range(len(angles)), key=lambda i: angles[i])
    return sum(distances[order[i]][order[(i + 1) % len(order)]] for i in range(len(order)))


def get_args():
    p = argparse.ArgumentParser(description='MPI TSP: Branch and Bound, dynamiczne taski')
    p.add_argument('--mode', choices=('mpi', 'serial'), default='mpi')
    p.add_argument('--n', type=int, default=13, help='liczba miast (zalecane 10-15)')
    p.add_argument('--depth', type=int, default=3, help='liczba miast prefiksu, wraz ze startem')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--radius', type=float, default=100.0)
    p.add_argument('--bound', choices=('minedge', 'degree'), default='minedge')
    p.add_argument('--output', default=None, help='sciezka do pliku JSON z wynikiem')
    p.add_argument('--tasks-output', default=None, help='opcjonalnie szczegoly taskow (JSONL)')
    args = p.parse_args()
    if not 3 <= args.n <= 20:
        p.error('--n musi byc miedzy 3 a 20')
    if not 1 <= args.depth <= args.n:
        p.error('--depth musi byc miedzy 1 a n')
    if args.radius <= 0:
        p.error('--radius musi byc dodatni')
    return args


def main():
    start = time.perf_counter()  # przed losowaniem miast i generowaniem taskow
    args = get_args()
    points, angles, distances = make_problem(args.n, args.seed, args.radius)
    before_prefixes = time.perf_counter()
    prefixes = generate_prefixes(args.n, args.depth)
    prefix_generation_s = time.perf_counter() - before_prefixes
    init_s = time.perf_counter() - start

    print(
        f'START N={args.n} seed={args.seed} d={args.depth} '
        f'taskow={len(prefixes)} bound={args.bound} mode={args.mode}',
        flush=True,
    )
    best_length = math.inf
    best_tour = None
    totals = {'nodes': 0, 'pruned': 0, 'leaves': 0, 'task_s': 0.0}
    tasks_per_rank = {}
    n_workers = 1

    def consume(iterator):
        nonlocal best_length, best_tour
        task_file = None
        try:
            if args.tasks_output:
                path = Path(args.tasks_output)
                path.parent.mkdir(parents=True, exist_ok=True)
                task_file = path.open('w', encoding='utf-8')
            for result in iterator:
                if task_file:
                    task_file.write(json.dumps(result) + '\n')
                rank = result['rank']
                r = tasks_per_rank.setdefault(rank, {'tasks': 0, 'nodes': 0, 'task_s': 0.0})
                r['tasks'] += 1
                r['nodes'] += result['nodes']
                r['task_s'] += result['task_s']
                for name in totals:
                    totals[name] += result[name]
                if result['best_cost'] < best_length:
                    best_length = result['best_cost']
                    best_tour = result['best_tour']
        finally:
            if task_file:
                task_file.close()

    if args.mode == 'serial':
        init_worker(distances, args.bound)
        consume(map(solve_subtree, prefixes))
    else:
        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor(initializer=init_worker, initargs=(distances, args.bound)) as executor:
            n_workers = executor.num_workers
            # Przydzial dynamiczny: kolejne prefiksy ida do wolnych workerow.
            consume(executor.map(solve_subtree, prefixes, chunksize=1))

    wall_s = time.perf_counter() - start
    expected = optimum_circle_length(angles, distances)
    correct = math.isclose(best_length, expected, rel_tol=1e-9, abs_tol=1e-7)
    if not correct:
        raise AssertionError(f'TSP: B&B={best_length:.12f}, optimum circle={expected:.12f}')

    summary = {
        'mode': args.mode,
        'n': args.n,
        'seed': args.seed,
        'radius': args.radius,
        'depth': args.depth,
        'bound': args.bound,
        'workers': n_workers,
        'mpi_processes': n_workers + 1 if args.mode == 'mpi' else 1,
        'task_count': len(prefixes),
        'prefix_generation_s': prefix_generation_s,
        'setup_including_prefixes_s': init_s,
        'wall_internal_s': wall_s,
        'best_cost': best_length,
        'best_tour': best_tour,
        'independent_optimal_length': expected,
        'verified': correct,
        'nodes': totals['nodes'],
        'pruned': totals['pruned'],
        'leaves': totals['leaves'],
        'sum_worker_task_s': totals['task_s'],
        'per_rank': tasks_per_rank,
        'points': [{'city': i + 1, 'x': x, 'y': y} for i, (x, y) in enumerate(points)],
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f'KONIEC p={n_workers} d={args.depth} T_internal={wall_s:.6f}s '
        f'generowanie_prefiksow={prefix_generation_s:.6f}s '
        f'best={best_length:.9f} wezly={totals["nodes"]} '
        f'pruned={totals["pruned"]} walidacja={correct}',
        flush=True,
    )
    print('TRASA:', ' -> '.join(map(str, best_tour)), flush=True)


if __name__ == '__main__':
    main()
