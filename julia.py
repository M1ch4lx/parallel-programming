#!/usr/bin/env python3

import argparse
import csv
import json
import time
from pathlib import Path

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

x0, x1, w = -2.0, +2.0, 640 * 2
y0, y1, h = -1.5, +1.5, 480 * 2
dx = (x1 - x0) / w
dy = (y1 - y0) / h
c = complex(0, 0.65)

SLEEP_LINE = None
SLEEP_SECONDS = 0.0


def configure_worker(sleep_line, sleep_seconds):
    global SLEEP_LINE, SLEEP_SECONDS
    SLEEP_LINE = sleep_line
    SLEEP_SECONDS = sleep_seconds


def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z ** 2 + c
        n -= 1
    return n


def julia_line(k):
    rank = MPI.COMM_WORLD.Get_rank()
    started = time.perf_counter()

    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)

    calc_s = time.perf_counter() - started
    slept_s = 0.0
    if k == SLEEP_LINE and SLEEP_SECONDS > 0.0:
        sleep_start = time.perf_counter()
        time.sleep(SLEEP_SECONDS)
        slept_s = time.perf_counter() - sleep_start

    total_s = time.perf_counter() - started

    # Wymagane w punkcie 1: print w julia_line, obejmuje kazda linie.
    print(
        f"linia={k:03d} rank={rank:02d} "
        f"czas_obliczen={calc_s:.6f}s "
        f"sleep={slept_s:.6f}s "
        f"czas_calkowity_linii={total_s:.6f}s",
        flush=True,
    )
    return k, bytes(line), rank, calc_s, slept_s, total_s


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 4 - zbior Julii, MPI Futures")
    parser.add_argument(
        "--mode", choices=("baseline", "sleep"), default="baseline",
        help="baseline: bez opoznienia; sleep: opoznij jedna linie",
    )
    parser.add_argument("--sleep-line", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--prefix", default=None,
        help="Prefiks plikow wyjsciowych (domyslnie baseline lub sleep)",
    )
    args = parser.parse_args()
    if args.mode == "sleep":
        if args.sleep_seconds <= 0:
            parser.error("--sleep-seconds musi byc > 0 w trybie sleep")
        if not 0 <= args.sleep_line < h:
            parser.error(f"--sleep-line musi nalezec do przedzialu 0..{h - 1}")
    return args


def main():
    args = parse_args()
    prefix = Path(args.prefix or args.mode)
    image_path = prefix.with_suffix(".pgm")
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")

    sleep_line = args.sleep_line if args.mode == "sleep" else None
    sleep_seconds = args.sleep_seconds if args.mode == "sleep" else 0.0
    started = time.perf_counter()

    print(
        f"START tryb={args.mode}; rozdzielczosc={w}x{h}; "
        f"sleep_line={sleep_line}; sleep_seconds={sleep_seconds}",
        flush=True,
    )

    with MPIPoolExecutor(
        initializer=configure_worker,
        initargs=(sleep_line, sleep_seconds),
    ) as executor:
        number_of_workers = number_of_workers = MPI.COMM_WORLD.Get_size() - 1
        with open(image_path, "wb") as image_file, open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("line", "rank", "calc_s", "sleep_s", "total_s"))
            image_file.write(b"P5 %d %d %d\n" % (w, h, 255))

            for expected_k, (k, pixels, rank, calc_s, slept_s, total_s) in enumerate(
                executor.map(julia_line, range(h), chunksize=1)
            ):
                if k != expected_k:
                    raise RuntimeError(f"Niepoprawna kolejnosc linii: {k} != {expected_k}")
                image_file.write(pixels)
                writer.writerow((k, rank, f"{calc_s:.9f}", f"{slept_s:.9f}", f"{total_s:.9f}"))

    wall_s = time.perf_counter() - started
    summary = {
        "mode": args.mode,
        "width": w,
        "height": h,
        "mpi_workers": number_of_workers,
        "sleep_line": sleep_line,
        "sleep_requested_s": sleep_seconds,
        "wall_s": wall_s,
        "csv": str(csv_path),
        "image": str(image_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"KONIEC: {args.mode}: {wall_s:.6f} s, workerow={number_of_workers}", flush=True)
    print(f"Zapisano: {image_path}, {csv_path}, {json_path}", flush=True)


if __name__ == "__main__":
    main()
