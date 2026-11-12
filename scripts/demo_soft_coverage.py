"""Task 6 demo: plot smooth vs hard coverage across increasing beta."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wsn.config import DEFAULT_CONFIG  # noqa: E402
from wsn.geometry import coverage_ratio, soft_coverage_ratio  # noqa: E402
from wsn.problem import build_problem  # noqa: E402


def main() -> None:
    config = DEFAULT_CONFIG.with_overrides(grid_resolution=60)
    problem = build_problem(config)

    rng = np.random.default_rng(config.seed)
    sensors = problem.random_layout(rng)
    rs = config.sensing_radius

    hard = coverage_ratio(sensors, problem.points, rs)
    betas = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400])
    soft = [soft_coverage_ratio(sensors, problem.points, rs, b) for b in betas]

    print(f"Hard (true) coverage: {hard:.4f}")
    for b, s in zip(betas, soft):
        print(f"  beta={b:6.1f}  soft={s:.4f}  |soft-hard|={abs(s-hard):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(betas, soft, "o-", label="soft coverage", lw=1.6)
    ax.axhline(hard, color="#e6550d", ls="--", label=f"hard coverage = {hard:.3f}")
    ax.set_xlabel("beta (sigmoid sharpness)")
    ax.set_ylabel("coverage ratio")
    ax.set_title("Soft coverage converges to hard coverage as beta grows")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(config.output_dir, exist_ok=True)
    out = os.path.join(config.output_dir, "task6_soft_vs_hard.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
