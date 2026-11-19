#!/usr/bin/env python3
"""Wykresy przyspieszenia, efektywnosci, serial fraction z pomiarow na Aresie."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Ares / wezly obliczeniowe bez X11
import matplotlib.pyplot as plt


SUMMARY_FIELDS = [
    'n', 'seed', 'bound', 'depth', 'workers', 'mpi_processes', 'task_count',
    'repeats', 'median_wall_s', 'min_wall_s', 'max_wall_s',
    'speedup', 'efficiency', 'serial_fraction', 'median_nodes', 'median_pruned',
]


def int_list(s):
    return sorted(set(int(v) for v in s.split(',')))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='measurements.csv')
    p.add_argument('--outdir', default=None)
    p.add_argument('--depth-fixed', type=int, default=3)
    p.add_argument('--workers', type=int_list, default=[1, 2, 4, 8])
    p.add_argument('--workers-fixed', type=int, default=8)
    p.add_argument('--depths', type=int_list, default=[2, 3, 4])
    return p.parse_args()


def fmt(x, digits=4):
    if x is None or not math.isfinite(x):
        return '—'
    return f'{x:.{digits}f}'


def main():
    args = parse_args()
    output = Path(args.outdir or Path(args.input).parent)
    output.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Brak pomiarow w CSV')
    problems = {(x['n'], x['seed'], x['radius'], x['bound']) for x in rows}
    if len(problems) != 1:
        raise ValueError('CSV ma rozne instancje TSP; wykresy wymagaja jednego problemu')
    n, seed, radius, bound = next(iter(problems))
    measurements = defaultdict(list)
    for row in rows:
        measurements[(int(row['depth']), int(row['workers']))].append(row)
    median_s = {
        key: statistics.median(float(r['wall_s']) for r in group)
        for key, group in measurements.items()
    }
    summary = []
    for (d, p), group in sorted(measurements.items()):
        t1 = median_s.get((d, 1))
        if t1 is None:
            raise ValueError(f'Brak pomiaru kontrolnego z 1 workerem przy d={d}')
        speedup = t1 / median_s[(d, p)]
        efficiency = speedup / p
        # Udzial sekwencyjny wg Karp-Flatt, dla p=1 wzor jest nieokreslony.
        serial = (1 / speedup - 1 / p) / (1 - 1 / p) if p > 1 else float('nan')
        summary.append({
            'n': n, 'seed': seed, 'bound': bound, 'depth': d,
            'workers': p, 'mpi_processes': p + 1,
            'task_count': int(group[0]['task_count']), 'repeats': len(group),
            'median_wall_s': median_s[(d, p)],
            'min_wall_s': min(float(r['wall_s']) for r in group),
            'max_wall_s': max(float(r['wall_s']) for r in group),
            'speedup': speedup, 'efficiency': efficiency,
            'serial_fraction': serial,
            'median_nodes': statistics.median(int(r['nodes']) for r in group),
            'median_pruned': statistics.median(int(r['pruned']) for r in group),
        })
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    lookup = {(x['depth'], x['workers']): x for x in summary}

    plots = [
        ('speedup', 'Przyspieszenie S(p) = T(1) / T(p)'),
        ('efficiency', 'Efektywnosc E(p) = S(p) / p'),
        ('serial_fraction', 'Serial fraction — estymator Karp-Flatt'),
    ]
    for experiment in (1, 2):
        if experiment == 1:
            fixed = args.depth_fixed
            requested = [(fixed, p) for p in args.workers]
            xlabel = 'Liczba workerow MPI (p)'
            name = f'eksperyment 1: stale d={fixed}, zmienne p'
        else:
            fixed = args.workers_fixed
            requested = [(d, fixed) for d in args.depths]
            xlabel = 'Glebokosc podzialu drzewa (d)'
            name = f'eksperyment 2: stale p={fixed}, zmienne d'
        for d, p in requested:
            if (d, p) not in lookup:
                raise ValueError(f'Nie znaleziono danych dla d={d}, p={p}')
        for key, ylabel in plots:
            points = []
            for d, p in requested:
                val = lookup[(d, p)][key]
                if math.isfinite(val):
                    points.append((p if experiment == 1 else d, val))
            fig, ax = plt.subplots(figsize=(8, 4.6))
            if points:
                xx, yy = zip(*points)
                ax.plot(xx, yy, 'o-', linewidth=2, markersize=6, label='Pomiar (mediana)')
            if experiment == 1:
                if key == 'speedup':
                    ax.plot(args.workers, args.workers, '--', alpha=0.6, label='Przyspieszenie idealne S=p')
                elif key == 'efficiency':
                    ax.axhline(1, linestyle='--', color='gray', alpha=0.6, label='Efektywnosc idealna E=1')
                else:
                    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
            elif key == 'serial_fraction':
                ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
            ax.set_xticks(args.workers if experiment == 1 else args.depths)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{name}\nN={n}, seed={seed}, bound={bound}')
            ax.grid(True, alpha=0.3)
            if experiment == 1 and key in ('speedup', 'efficiency'):
                ax.legend(loc='best')
            fig.tight_layout()
            namefile = output / f'exp{experiment}_{key}.png'
            fig.savefig(namefile, dpi=180)
            plt.close(fig)
            print('Zapisano', namefile)

    # Raport liczbowy, bez automatycznego domyslania sie wnioskow z wykresow.
    report = [
        '# Rownolegly Branch and Bound — TSP na okregu',
        '',
        f'Instancja: N={n}, seed={seed}, promien={radius}, bound={bound}.',
        'Czas T jest czasem zewnetrznym subprocess, tzn. obejmuje uruchomienie MPI,',
        'przygotowanie danych, generowanie wszystkich prefiksow, obliczenia i zamkniecie MPI.',
        'T(p) jest mediana z powtorzen. p oznacza workery, poza nimi istnieje proces master.',
        '',
        '## Eksperyment 1 — stale d, zmienna liczba workerow',
        '',
        '| d | p | taski | T(p) [s] | S(p) | E(p) | serial fraction |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for d, p in [(args.depth_fixed, p) for p in args.workers]:
        r = lookup[(d, p)]
        report.append(f'| {d} | {p} | {r["task_count"]} | {fmt(r["median_wall_s"])} | '
                      f'{fmt(r["speedup"])} | {fmt(r["efficiency"])} | {fmt(r["serial_fraction"])} |')
    report += [
        '', '## Eksperyment 2 — stale p, zmienna glebokosc d', '',
        '| d | p | taski | T1(d) [s] | Tp(d) [s] | S(p,d) | E(p,d) | serial fraction |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for d in args.depths:
        r = lookup[(d, args.workers_fixed)]
        t1 = lookup[(d, 1)]['median_wall_s']
        report.append(f'| {d} | {args.workers_fixed} | {r["task_count"]} | {fmt(t1)} | '
                      f'{fmt(r["median_wall_s"])} | {fmt(r["speedup"])} | '
                      f'{fmt(r["efficiency"])} | {fmt(r["serial_fraction"])} |')
    report += [
        '', '## Wzory i interpretacja', '',
        '- Przyspieszenie S(p,d) = T(1,d) / T(p,d).',
        '- Efektywnosc E(p,d) = S(p,d) / p.',
        '- Serial fraction f(p,d) = (1/S(p,d) - 1/p) / (1 - 1/p), tylko p > 1.',
        '- Glebokosc d zmienia nie tylko liczbe taskow, ale i skutecznosc lokalnego pruningu;',
        '  T(1,d) moze byc rozne dla roznych d. Z tego powodu porownujemy z bazowym T(1,d).',
        '- Ujemna serial fraction moze wystapic przy przyspieszeniu superliniowym lub szumie pomiarowym.',
        '- Dla p=1 serial fraction jest nieokreslona (myslnik w tabeli).',
        '', '## PCAM', '',
        '1. Partitioning: wszystkie prefiksy od miasta 1 do glebokosci d.',
        '2. Communication: master przesyla prefiks; worker odsyla koszt i trase;',
        '   brak wymiany lokalnych gornych granic miedzy workerami.',
        '3. Agglomeration: jeden task = cale poddrzewo jednego prefiksu; chunksize=1.',
        '4. Mapping: dynamiczny przydzial wolnym workerom przez MPIPoolExecutor.map.',
        '', '## Pliki', '',
        'measurements.csv — surowe pomiary; summary.csv — mediany i metryki;',
        'exp1_*.png i exp2_*.png — szesc wymaganych wykresow;',
        'runs/*.json — wyniki TSP; logs/*.log — logi uruchomien MPI.',
    ]
    (output / 'raport.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('Zapisano', output / 'summary.csv', 'oraz', output / 'raport.md')


if __name__ == '__main__':
    main()
