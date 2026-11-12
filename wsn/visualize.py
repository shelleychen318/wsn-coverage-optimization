"""Visualization helpers: coverage maps, convergence curves, comparison tables.

Uses the non-interactive Agg backend so scripts save figures without opening
a window (safe to run headless / from the command line).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .problem import Problem  # noqa: E402


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_coverage_map(
    problem: Problem,
    sensors: np.ndarray,
    *,
    title: str = "Coverage map",
    show_edges: bool = False,
    edges: Optional[Sequence[tuple]] = None,
    save_path: Optional[str] = None,
):
    """Render sampled points colored by coverage, with sensing circles.

    Parameters
    ----------
    show_edges, edges
        If ``show_edges`` is True, draw communication edges. ``edges`` is an
        optional explicit list of (i, j) index pairs into the node array
        ``[sensors; base_station]``; if omitted the caller is expected to pass
        edges in later tasks. Kept optional so Task 2 can ignore connectivity.
    """
    from .geometry import coverage_mask

    rs = problem.sensing_radius
    pts = problem.points
    mask = coverage_mask(sensors, pts, rs)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Covered vs uncovered sampled points.
    ax.scatter(
        pts[~mask, 0], pts[~mask, 1], s=6, c="#d9d9d9", label="uncovered", zorder=1
    )
    ax.scatter(
        pts[mask, 0], pts[mask, 1], s=6, c="#9ecae1", label="covered", zorder=1
    )

    # Sensing circles.
    for sx, sy in sensors:
        circ = plt.Circle(
            (sx, sy), rs, fill=False, color="#3182bd", lw=1.0, alpha=0.8, zorder=2
        )
        ax.add_patch(circ)

    # Communication edges (optional, used from Task 3 onward).
    if show_edges:
        if edges is None:
            from .geometry import comm_edges

            edges = comm_edges(sensors, problem.base_station, problem.comm_radius)
        nodes = np.vstack([sensors, problem.base_station[None, :]])
        for i, j in edges:
            ax.plot(
                [nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                color="#fdae6b",
                lw=0.8,
                alpha=0.7,
                zorder=3,
            )

    # Sensors and base station.
    ax.scatter(
        sensors[:, 0], sensors[:, 1], s=40, c="#08519c", marker="^",
        label="sensors", zorder=4,
    )
    bx, by = problem.base_station
    ax.scatter(
        [bx], [by], s=120, c="#e6550d", marker="*", label="base station", zorder=5
    )

    x_min, y_min, x_max, y_max = problem.region
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax


def plot_convergence(
    curves: dict,
    *,
    title: str = "Convergence",
    ylabel: str = "Best coverage ratio",
    save_path: Optional[str] = None,
):
    """Overlay one or more convergence curves.

    Parameters
    ----------
    curves
        Mapping of label -> sequence of per-iteration values.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, values in curves.items():
        ax.plot(range(len(values)), values, label=label, lw=1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax


def plot_seed_spread(
    coverages_by_method: dict,
    *,
    title: str = "Sensitivity to initial conditions",
    save_path: Optional[str] = None,
):
    """Box + strip plot of final coverage across random starts, per method.

    A tight box => robust to the starting layout; a wide box with low outliers
    => sensitive to initialization (and possibly to infeasible local optima).
    """
    methods = list(coverages_by_method.keys())
    data = [np.asarray(coverages_by_method[m], dtype=float) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=methods, showmeans=True)
    # Overlay individual points.
    for i, vals in enumerate(data, start=1):
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.12
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.6, c="#3182bd")
    ax.set_ylabel("Final coverage ratio")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax


def plot_parameter_sweep(
    sweep_points,
    *,
    metric: str = "mean_coverage",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Line plot of a swept parameter vs a chosen metric, one line per method.

    ``sweep_points`` is the list of SweepPoint objects from
    :func:`wsn.experiment.sweep_parameter`.
    """
    methods = sorted({p.method for p in sweep_points})
    param_name = sweep_points[0].param_name if sweep_points else "param"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        pts = sorted(
            [p for p in sweep_points if p.method == method], key=lambda p: p.param_value
        )
        xs = [p.param_value for p in pts]
        ys = [getattr(p, metric) for p in pts]
        ax.plot(xs, ys, "o-", label=method, lw=1.6)

    ax.set_xlabel(param_name)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"{metric.replace('_', ' ')} vs {param_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax


def plot_comparison_table(
    summaries: dict,
    *,
    title: str = "Method comparison",
    save_path: Optional[str] = None,
):
    """Render the method-comparison summary as a figure (table of metrics).

    ``summaries`` is a mapping method -> MethodSummary (from the experiment
    runner). Columns cover all five comparison axes from the proposal:
    coverage (best/mean/worst), constraint satisfaction, and compute cost.
    """
    methods = list(summaries.keys())
    col_labels = [
        "best C", "mean C", "worst C", "std C",
        "feasible %", "mean time (s)", "mean evals",
    ]
    rows = []
    for m in methods:
        s = summaries[m]
        rows.append(
            [
                f"{s.best_coverage:.3f}",
                f"{s.mean_coverage:.3f}",
                f"{s.worst_coverage:.3f}",
                f"{s.std_coverage:.3f}",
                f"{100 * s.feasibility_rate:.0f}",
                f"{s.mean_time:.2f}",
                f"{s.mean_evaluations:.0f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * len(methods)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=methods,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title(title, pad=20)

    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax


def _resample(values, n):
    """Linearly resample a 1-D sequence to length ``n`` (for curve overlay)."""
    values = np.asarray(values, dtype=float)
    if len(values) == n:
        return values
    if len(values) == 1:
        return np.repeat(values, n)
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, n)
    return np.interp(new_x, old_x, values)


def plot_best_convergence(
    summaries: dict,
    *,
    title: str = "Best-run convergence (normalized progress)",
    save_path: Optional[str] = None,
):
    """Overlay the best run's convergence curve from each method.

    PSO logs per-iteration best coverage; SQP logs per-continuation-stage true
    coverage. The x-axis is normalized progress (0..1) so the differently
    lengthed histories can be compared on one plot.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Longest history sets the resampling resolution.
    n = max(len(s.best_record.convergence) for s in summaries.values())
    n = max(n, 2)
    x = np.linspace(0.0, 1.0, n)
    for method, s in summaries.items():
        y = _resample(s.best_record.convergence, n)
        ax.plot(x, y, "o-", ms=3, label=f"{method} (best)", lw=1.6)
    ax.set_xlabel("normalized optimization progress")
    ax.set_ylabel("coverage ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or ".")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax
