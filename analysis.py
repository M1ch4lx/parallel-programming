#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(filename):
    with open(filename, newline="", encoding="utf-8") as f:
        records = [
            {
                "line": int(r["line"]),
                "rank": int(r["rank"]),
                "calc_s": float(r["calc_s"]),
                "sleep_s": float(r["sleep_s"]),
                "total_s": float(r["total_s"]),
            }
            for r in csv.DictReader(f)
        ]
    if len(records) != 960 or sorted(r["line"] for r in records) != list(range(960)):
        raise ValueError(f"{filename}: oczekiwano 960 unikalnych linii od 0 do 959")
    records.sort(key=lambda r: r["line"])
    return records


def per_worker(records):
    result = defaultdict(lambda: {"lines": [], "calc_s": 0.0, "sleep_s": 0.0, "total_s": 0.0})
    for r in records:
        entry = result[r["rank"]]
        entry["lines"].append(r["line"])
        for key in ("calc_s", "sleep_s", "total_s"):
            entry[key] += r[key]
    return dict(sorted(result.items()))


def regular_mapping(records, worker_count):
    ranks = [r["rank"] for r in records]
    groups = per_worker(records)
    
    block = all(lines == list(range(lines[0], lines[-1] + 1)) for lines in
                (d["lines"] for d in groups.values()) if lines)
    
    first = ranks[:worker_count]
    cyclic = (
        len(first) == worker_count
        and len(set(first)) == worker_count
        and all(rank == first[i % worker_count] for i, rank in enumerate(ranks))
    )
    return block, cyclic


def one_run_report(name, records, metadata):
    groups = per_worker(records)
    min_r = min(records, key=lambda r: r["calc_s"])
    max_r = max(records, key=lambda r: r["calc_s"])
    delta = max_r["calc_s"] - min_r["calc_s"]
    wall_s = float(metadata["wall_s"])
    worker_count = int(metadata["mpi_workers"])
    block, cyclic = regular_mapping(records, worker_count)

    out = [
        "=" * 75,
        f"PRZEBIEG {name.upper()} - {len(records)} linii, {worker_count} workerow",
        f"Calkowity czas programu (wall time, wl. MPI i zapis pliku): {wall_s:.6f} s",
        "",
        "Punkt 2. Zmiennosc czasow obliczania linii (BEZ sleep):",
        f"  Minimum: linia {min_r['line']}, rank {min_r['rank']}, {min_r['calc_s']:.6f} s",
        f"  Maksimum: linia {max_r['line']}, rank {max_r['rank']}, {max_r['calc_s']:.6f} s",
        f"  Roznica max-min: {delta:.6f} s",
        "",
        "Punkt 3. Przydzial linii do rang MPI:",
        "  Pierwsze 40 przypisanych rang (kolejnosc numerow linii):",
        "  " + " ".join(f"{r['rank']:02d}" for r in records[:40]),
        f"  Scisle blokowy: {'TAK' if block else 'NIE'}",
        f"  Scisle cykliczny: {'TAK' if cyclic else 'NIE'}",
        "  Algorytm: dynamiczna kolejka zadan, wolny worker dostaje kolejna linie.",
        "  UWAGA: chwila podobienstwa do cyklicznego nie oznacza statycznego przydzialu.",
        "",
        "Punkty 4 i 5. Liczba linii, suma czasow, porownanie z calym programem:",
        "  rank | linie | suma obliczen [s] | suma sleep [s] | suma razem [s] | razem / wall",
    ]
    for rank, g in groups.items():
        fraction = g["total_s"] / wall_s if wall_s > 0 else 0
        out.append(
            f"  {rank:4d} | {len(g['lines']):5d} | {g['calc_s']:17.6f} |"
            f" {g['sleep_s']:14.6f} | {g['total_s']:14.6f} | {fraction:11.2%}"
        )

    counts = [len(g["lines"]) for g in groups.values()]
    times = [g["total_s"] for g in groups.values()]
    out += [
        f"  Roznica liczby linii max-min: {max(counts) - min(counts)}",
        f"  Roznica sum czasow workerow max-min: {max(times) - min(times):.6f} s",
        f"  Suma czasow obliczen wszystkich workerow: {sum(g['calc_s'] for g in groups.values()):.6f} s",
        f"  Suma czasow razem wszystkich workerow: {sum(times):.6f} s",
        "  Sumy workerow moga przekraczac wall time, gdyz procesy licza rownolegle.",
        "  Wall time dodatkowo obejmuje start MPI, komunikacje i zapis obrazu.",
    ]
    return "\n".join(out)


def compare_report(a, b, a_meta, b_meta):
    ga, gb = per_worker(a), per_worker(b)
    wa, wb = float(a_meta["wall_s"]), float(b_meta["wall_s"])
    output = [
        "=" * 75,
        "Punkt 6. POROWNANIE PRZED / PO SLEEP",
        f"Sleep ustawiony dla linii {b_meta['sleep_line']}, "
        f"zadany czas {b_meta['sleep_requested_s']:.3f} s.",
        f"Czas programu: {wa:.6f} s -> {wb:.6f} s "
        f"(roznica {wb - wa:+.6f} s).",
        "",
        "rank | liczba linii: przed -> po | suma czasu linii [s]: przed -> po",
    ]
    for rank in sorted(set(ga) | set(gb)):
        pa = ga.get(rank, {"lines": [], "total_s": 0.0})
        pb = gb.get(rank, {"lines": [], "total_s": 0.0})
        output.append(
            f"{rank:4d} | {len(pa['lines']):4d} -> {len(pb['lines']):4d} |"
            f" {pa['total_s']:10.5f} -> {pb['total_s']:10.5f}"
        )
    slow_line = int(b_meta["sleep_line"])
    sa = next(r for r in a if r["line"] == slow_line)
    sb = next(r for r in b if r["line"] == slow_line)
    output += [
        "",
        f"Linia {slow_line}: rank przed={sa['rank']}, rank po={sb['rank']}, "
        f"faktyczny sleep={sb['sleep_s']:.6f} s.",
        "Wniosek do weryfikacji: worker obciazony sleep dostaje zwykle mniej "
        "kolejnych linii, poniewaz inne procesy odbieraja zadania z kolejki.",
        "Obserwacje opieraj na liczbach z tabeli; rozklad jest niedeterministyczny.",
        "",
        "Punkt 7. PCAM:",
        "P - Partitioning: obraz rozbijany na 960 niezaleznych zadan (po jednej linii).",
        "C - Communication: master wysyla numer linii; worker zwraca 1280 bajtow "
        "pikseli i metadane (rank, czas). Brak wymiany miedzy workerami.",
        "A - Agglomeration: granulat 1 linia (chunksize=1); prostota i "
        "rownomierny podzial kosztem 960 komunikatow/zadan.",
        "M - Mapping: dynamiczna kolejka MPIPoolExecutor; gdy worker konczy "
        "linie, otrzymuje nastepna. Sleep testuje rownowazenie obciazenia.",
        "Potencjalne ograniczenia: narzut MPI, rozne koszty linii, "
        "szeregowy zapis PGM przez mastera, efekty oczekiwania na wyniki in-order.",
    ]
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    recomm = sub.add_parser("recommend", help="Dobierz sleep >= maks. sumy obliczen workera")
    recomm.add_argument("--file", default="baseline.csv")
    report = sub.add_parser("report", help="Wypisz raport punktow 2-7")
    report.add_argument("--base", default="baseline.csv")
    report.add_argument("--slow", default="sleep.csv")
    report.add_argument("--base-meta", default="baseline.json")
    report.add_argument("--slow-meta", default="sleep.json")
    args = parser.parse_args()

    if args.command == "recommend":
        data = per_worker(read_csv(args.file))
        slow = max(1.0, max(g["calc_s"] for g in data.values()) * 1.25)
        print(f"{slow:.3f}")  # Tylko liczba: uruchomienie przez bash $(...).
    else:
        a = read_csv(args.base)
        b = read_csv(args.slow)
        a_meta = json.loads(Path(args.base_meta).read_text(encoding="utf-8"))
        b_meta = json.loads(Path(args.slow_meta).read_text(encoding="utf-8"))
        print("LAB 4 - ANALIZA ZBIORU JULII (MPI FUTURES)")
        print(one_run_report("baseline", a, a_meta))
        print()
        print(one_run_report("sleep", b, b_meta))
        print()
        print(compare_report(a, b, a_meta, b_meta))


if __name__ == "__main__":
    main()
