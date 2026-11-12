"""Multi-start experiment runner and metric collection.

Runs PSO and SQP from several seeded random starts and collects, for each run:
  * final coverage ratio (the TRUE Boolean metric),
  * constraint satisfaction (TRUE BFS connectivity check, not the surrogate),
  * wall-clock time,
  * per-iteration / per-stage convergence history.

Results are plain dataclasses so they are easy to aggregate, print as a
table, or feed into the visualization module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .geometry import coverage_ratio, is_connected
from .problem import Problem, build_problem
from .pso import solve_pso
from .sqp import run_sqp


@dataclass
class RunRecord:
    """One solver run from one random start."""

    method: str  # 'PSO-penalty', 'PSO-repair', or 'SQP'
    seed: int
    coverage: float  # true coverage ratio
    connected: bool  # true BFS feasibility
    time_sec: float
    n_evaluations: int
    convergence: List[float] = field(default_factory=list)  # coverage per step
    layout: Optional[np.ndarray] = None


@dataclass
class MethodSummary:
    method: str
    best_coverage: float
    mean_coverage: float
    worst_coverage: float
    std_coverage: float
    feasibility_rate: float  # fraction of runs that are truly connected
    mean_time: float
    mean_evaluations: float
    best_record: RunRecord
    records: List[RunRecord] = field(default_factory=list)


def _run_one(method: str, problem: Problem, config: Config, seed: int) -> RunRecord:
    """Run a single solver from a single seed and package the metrics."""
    rng = np.random.default_rng(seed)
    x0 = problem.random_layout(rng)  # shared start for fairness across methods

    t0 = time.perf_counter()
    if method == "SQP":
        res = run_sqp(problem, config, seed=seed, x0=x0)
        convergence = res.history
        layout = res.best_layout
        n_evals = res.n_evaluations
    elif method in ("PSO-penalty", "PSO-repair"):
        mode = "penalty" if method == "PSO-penalty" else "repair"
        cfg = config.with_overrides(pso_constraint_mode=mode)
        # PSO initializes its own swarm; seed controls reproducibility.
        res = solve_pso(problem, cfg, seed=seed)
        convergence = res.coverage_history
        layout = res.best_layout
        n_evals = res.n_evaluations
    else:
        raise ValueError(f"unknown method: {method}")
    elapsed = time.perf_counter() - t0

    # Always validate with the TRUE checks, independent of solver internals.
    true_cov = coverage_ratio(layout, problem.points, problem.sensing_radius)
    connected = is_connected(layout, problem.base_station, problem.comm_radius)

    return RunRecord(
        method=method,
        seed=seed,
        coverage=true_cov,
        connected=connected,
        time_sec=elapsed,
        n_evaluations=n_evals,
        convergence=list(convergence),
        layout=layout,
    )


def _summarize(method: str, records: List[RunRecord]) -> MethodSummary:
    covs = np.array([r.coverage for r in records])
    best_idx = int(np.argmax(covs))
    return MethodSummary(
        method=method,
        best_coverage=float(np.max(covs)),
        mean_coverage=float(np.mean(covs)),
        worst_coverage=float(np.min(covs)),
        std_coverage=float(np.std(covs)),
        feasibility_rate=float(np.mean([r.connected for r in records])),
        mean_time=float(np.mean([r.time_sec for r in records])),
        mean_evaluations=float(np.mean([r.n_evaluations for r in records])),
        best_record=records[best_idx],
        records=records,
    )


def run_experiment(
    config: Optional[Config] = None,
    methods: Optional[List[str]] = None,
    problem: Optional[Problem] = None,
) -> Dict[str, MethodSummary]:
    """Run all methods from ``config.n_starts`` seeded random starts.

    Returns a mapping method -> :class:`MethodSummary`.
    """
    config = config or Config()
    config.validate()
    problem = problem or build_problem(config)
    methods = methods or ["PSO-penalty", "PSO-repair", "SQP"]

    seeds = [config.seed + i for i in range(config.n_starts)]
    summaries: Dict[str, MethodSummary] = {}
    for method in methods:
        records = [_run_one(method, problem, config, s) for s in seeds]
        summaries[method] = _summarize(method, records)
    return summaries


def format_summary_table(summaries: Dict[str, MethodSummary]) -> str:
    """Render a plain-text comparison table across methods."""
    header = (
        f"{'method':<14}{'best':>8}{'mean':>8}{'worst':>8}{'std':>8}"
        f"{'feas%':>8}{'time(s)':>10}{'evals':>10}"
    )
    lines = [header, "-" * len(header)]
    for method, s in summaries.items():
        lines.append(
            f"{method:<14}{s.best_coverage:>8.3f}{s.mean_coverage:>8.3f}"
            f"{s.worst_coverage:>8.3f}{s.std_coverage:>8.3f}"
            f"{100 * s.feasibility_rate:>8.1f}{s.mean_time:>10.3f}"
            f"{s.mean_evaluations:>10.0f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sensitivity sweeps (Task 9)
# ---------------------------------------------------------------------------
@dataclass
class SweepPoint:
    """One (parameter-value, method) aggregate across the seeded starts."""

    param_name: str
    param_value: float
    method: str
    mean_coverage: float
    std_coverage: float
    best_coverage: float
    feasibility_rate: float
    coverages: List[float] = field(default_factory=list)  # per-seed values


def sweep_initial_conditions(
    config: Optional[Config] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """Return, per method, the list of final coverages across all random starts.

    This isolates the "sensitivity to initial conditions" question: with the
    same problem and the same set of seeds, how widely do the methods' outcomes
    spread? A tight spread => robust to the start; a wide spread => sensitive.
    """
    config = config or Config()
    summaries = run_experiment(config, methods=methods)
    return {m: [r.coverage for r in s.records] for m, s in summaries.items()}


def sweep_parameter(
    param_name: str,
    values,
    base_config: Optional[Config] = None,
    methods: Optional[List[str]] = None,
) -> List[SweepPoint]:
    """Sweep a single Config field over ``values`` and aggregate per method.

    Example: ``sweep_parameter('n_sensors', [6, 8, 10])``.
    """
    base_config = base_config or Config()
    methods = methods or ["PSO-penalty", "PSO-repair", "SQP"]
    results: List[SweepPoint] = []

    for value in values:
        cfg = base_config.with_overrides(**{param_name: value})
        # Rebuild the problem since structural params (N, radii, grid) change it.
        summaries = run_experiment(cfg, methods=methods)
        for method, s in summaries.items():
            covs = [r.coverage for r in s.records]
            results.append(
                SweepPoint(
                    param_name=param_name,
                    param_value=float(value),
                    method=method,
                    mean_coverage=s.mean_coverage,
                    std_coverage=s.std_coverage,
                    best_coverage=s.best_coverage,
                    feasibility_rate=s.feasibility_rate,
                    coverages=covs,
                )
            )
    return results
