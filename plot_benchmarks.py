from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["n"] = int(row["n"])
        row["processes"] = int(row["processes"])
        row["time_seconds"] = float(row["time_seconds"])
    return rows


def metrics_by_size(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["n"], []).append(row)

    for n, items in grouped.items():
        items.sort(key=lambda r: r["processes"])
        baseline = next((r for r in items if r["processes"] == 1), None)
        if baseline is None:
            raise ValueError(f"Missing P=1 baseline for N={n}.")
        t1 = baseline["time_seconds"]

        for row in items:
            p = row["processes"]
            speedup = t1 / row["time_seconds"]
            efficiency = speedup / p
            karp_flatt = math.nan if p == 1 else ((1.0 / speedup) - (1.0 / p)) / (1.0 - 1.0 / p)
            row["speedup"] = speedup
            row["efficiency"] = efficiency
            row["karp_flatt"] = karp_flatt

    return grouped


def make_plot(grouped: dict[int, list[dict]], metric: str, ylabel: str, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for n in sorted(grouped):
        items = grouped[n]
        xs = [r["processes"] for r in items]
        ys = [r[metric] for r in items]
        ax.plot(xs, ys, marker="o", label=f"N={n} (siatka {n}×{n})")

    all_p = sorted({r["processes"] for items in grouped.values() for r in items})
    ax.set_xticks(all_p)
    ax.set_xlabel("Liczba procesów P")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot S(P), E(P), and Karp-Flatt f(P).")
    parser.add_argument("--input", type=Path, default=here / "benchmark.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "plots")
    args = parser.parse_args()

    rows = load_rows(args.input)
    grouped = metrics_by_size(rows)
    if len(grouped) < 2:
        print("WARNING: assignment asks for at least two problem sizes; current CSV contains fewer than two.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_plot(
        grouped,
        "speedup",
        "Przyspieszenie S(P) = T(1) / T(P)",
        "Przyspieszenie algorytmu równoległego",
        args.output_dir / "speedup.png",
    )
    make_plot(
        grouped,
        "efficiency",
        "Efektywność E(P) = S(P) / P",
        "Efektywność algorytmu równoległego",
        args.output_dir / "efficiency.png",
    )
    make_plot(
        grouped,
        "karp_flatt",
        "Metryka Karpa–Flatta f(P)",
        "Metryka Karpa–Flatta",
        args.output_dir / "karp_flatt.png",
    )

    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
