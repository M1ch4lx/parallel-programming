from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class SolverResult:
    field: np.ndarray
    iterations: int
    max_update: float
    elapsed: float


def validate_problem(n: int, a: float, p: float, tension: float) -> None:
    if n < 3:
        raise ValueError("n must be at least 3 (two boundary points + interior).")
    if a <= 0:
        raise ValueError("a must be > 0.")
    if p < 0:
        raise ValueError("p must be >= 0.")
    if tension <= 0:
        raise ValueError("T (tension) must be > 0.")


def jacobi_serial(
    n: int,
    a: float,
    p: float,
    tension: float,
    tol: float = 1.0e-8,
    max_iters: int = 200_000,
) -> SolverResult:
    """Solve Δh = -p/T on [0,a]x[0,a], with h=0 on the boundary.

    A uniform n x n grid is used. Jacobi iteration gives

        h[i,j] = 1/4 * (h[i-1,j] + h[i+1,j]
                        + h[i,j-1] + h[i,j+1] + dx^2 p/T).

    The sign therefore yields positive deflection h for p>0.
    """
    validate_problem(n, a, p, tension)
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1.")

    old = np.zeros((n, n), dtype=np.float64)
    new = np.zeros_like(old)
    dx = a / (n - 1)
    rhs_term = dx * dx * p / tension

    t0 = perf_counter()
    max_update = float("inf")

    for iteration in range(1, max_iters + 1):
        new[1:-1, 1:-1] = 0.25 * (
            old[:-2, 1:-1]
            + old[2:, 1:-1]
            + old[1:-1, :-2]
            + old[1:-1, 2:]
            + rhs_term
        )

        max_update = float(np.max(np.abs(new[1:-1, 1:-1] - old[1:-1, 1:-1])))
        old, new = new, old

        if max_update < tol:
            break

    elapsed = perf_counter() - t0
    return SolverResult(old.copy(), iteration, max_update, elapsed)


def save_visualizations(field: np.ndarray, a: float, prefix: str | Path) -> tuple[Path, Path]:
    """Save heatmap and 3D surface of the membrane deflection."""
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path = prefix.with_name(prefix.name + "_heatmap.png")
    surface_path = prefix.with_name(prefix.name + "_surface3d.png")

    n = field.shape[0]
    x = np.linspace(0.0, a, n)
    y = np.linspace(0.0, a, n)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(
        field,
        origin="lower",
        extent=(0.0, a, 0.0, a),
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="ugięcie h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ugięcie kwadratowej membrany — mapa ciepła")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=170)
    plt.close(fig)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, field, linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("h")
    ax.set_title("Ugięcie kwadratowej membrany — powierzchnia 3D")
    fig.tight_layout()
    fig.savefig(surface_path, dpi=170)
    plt.close(fig)

    return heatmap_path, surface_path
