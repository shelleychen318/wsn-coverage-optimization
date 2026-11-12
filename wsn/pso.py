"""Hand-written Particle Swarm Optimization for sensor placement.

Each particle encodes a full sensor layout flattened to a vector of length
2N (``[x0, y0, x1, y1, ...]``). The swarm maximizes a fitness equal to the
true coverage ratio minus a penalty for violating the connectivity constraint
(penalty mode), or with disconnected layouts repaired before evaluation
(repair mode, added in Task 5).

The implementation is intentionally explicit (no third-party PSO library) so
the constraint handling, convergence logging, and seeding are fully visible
for the comparison and writeup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .config import Config
from .geometry import algebraic_connectivity, coverage_ratio, is_connected
from .problem import Problem


def solve_pso(problem: Problem, config: Optional[Config] = None, seed: Optional[int] = None):
    """Run PSO using the constraint mode selected in the config.

    'penalty' -> penalize disconnected layouts in the fitness.
    'repair'  -> repair disconnected layouts before evaluation.
    """
    config = config or problem.config
    if config.pso_constraint_mode == "repair":
        return run_pso(problem, config, seed=seed, repair_fn=repair_layout)
    return run_pso(problem, config, seed=seed)


@dataclass
class PSOResult:
    """Outcome of a single PSO run."""

    best_layout: np.ndarray  # (N, 2)
    best_fitness: float
    best_coverage: float
    connected: bool
    history: List[float] = field(default_factory=list)  # best fitness per iter
    coverage_history: List[float] = field(default_factory=list)
    n_evaluations: int = 0


def _disconnection_penalty(
    sensors: np.ndarray, problem: Problem, weight: float
) -> float:
    """Smooth, non-negative penalty that grows as the graph loses connectivity.

    Uses ``max(0, eps - lambda_2)`` with a sharp beta so a well-connected
    layout pays ~0 while a disconnected one pays up to ``weight * eps``. This
    keeps the penalty bounded and comparable in scale to the coverage ratio.
    """
    base, rc = problem.base_station, problem.comm_radius
    eps = problem.config.lambda2_epsilon
    beta = problem.config.beta_schedule[-1]
    lam2 = algebraic_connectivity(sensors, base, rc, beta=beta)
    shortfall = max(0.0, eps - lam2) if lam2 < eps else 0.0
    # Hard component: if truly disconnected, apply the full weight so the
    # swarm strongly prefers feasible layouts.
    hard = 0.0 if is_connected(sensors, base, rc) else 1.0
    return weight * (hard + shortfall)


def _fitness(sensors: np.ndarray, problem: Problem, config: Config) -> tuple:
    """Return (fitness, coverage) for a layout under penalty constraint mode."""
    cov = coverage_ratio(sensors, problem.points, problem.sensing_radius)
    penalty = _disconnection_penalty(sensors, problem, config.pso_penalty_weight)
    return cov - penalty, cov


def run_pso(
    problem: Problem,
    config: Optional[Config] = None,
    seed: Optional[int] = None,
    repair_fn: Optional[Callable[[np.ndarray, Problem], np.ndarray]] = None,
) -> PSOResult:
    """Run PSO and return the best layout found.

    Parameters
    ----------
    repair_fn
        Optional repair operator (used in Task 5 'repair' mode). If provided,
        each candidate is repaired before fitness evaluation and the penalty
        term is skipped.
    """
    config = config or problem.config
    seed = config.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    n = problem.n_sensors
    dim = 2 * n
    x_min, y_min, x_max, y_max = config.region
    lower = np.array([x_min, y_min] * n)
    upper = np.array([x_max, y_max] * n)

    # Velocity clamp per axis.
    vmax = config.pso_velocity_clamp_frac * np.array(
        [config.region_width, config.region_height] * n
    )

    # --- initialize swarm ---------------------------------------------------
    pos = rng.uniform(lower, upper, size=(config.pso_swarm_size, dim))
    vel = rng.uniform(-vmax, vmax, size=(config.pso_swarm_size, dim))

    use_repair = repair_fn is not None
    n_evals = 0

    def evaluate(vec: np.ndarray) -> tuple:
        nonlocal n_evals
        n_evals += 1
        layout = vec.reshape(n, 2)
        if use_repair:
            layout = repair_fn(layout, problem)
            cov = coverage_ratio(layout, problem.points, problem.sensing_radius)
            fit = cov  # repaired -> feasible, no penalty
            return fit, cov, layout
        fit, cov = _fitness(layout, problem, config)
        return fit, cov, layout

    pbest_pos = pos.copy()
    pbest_fit = np.full(config.pso_swarm_size, -np.inf)
    pbest_cov = np.zeros(config.pso_swarm_size)

    for i in range(config.pso_swarm_size):
        fit, cov, layout = evaluate(pos[i])
        pbest_fit[i] = fit
        pbest_cov[i] = cov
        if use_repair:
            # Store the repaired layout as the particle position.
            pos[i] = layout.reshape(-1)
            pbest_pos[i] = pos[i]

    g_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])
    gbest_cov = float(pbest_cov[g_idx])

    history = [gbest_fit]
    coverage_history = [gbest_cov]

    w, c1, c2 = config.pso_inertia, config.pso_cognitive, config.pso_social

    # --- main loop ----------------------------------------------------------
    for _ in range(config.pso_iterations):
        r1 = rng.random((config.pso_swarm_size, dim))
        r2 = rng.random((config.pso_swarm_size, dim))
        vel = (
            w * vel
            + c1 * r1 * (pbest_pos - pos)
            + c2 * r2 * (gbest_pos - pos)
        )
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        pos = np.clip(pos, lower, upper)  # stay inside M

        for i in range(config.pso_swarm_size):
            fit, cov, layout = evaluate(pos[i])
            if use_repair:
                pos[i] = layout.reshape(-1)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_cov[i] = cov
                pbest_pos[i] = pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit = float(fit)
                    gbest_cov = float(cov)
                    gbest_pos = pos[i].copy()

        history.append(gbest_fit)
        coverage_history.append(gbest_cov)

    best_layout = gbest_pos.reshape(n, 2)
    connected = is_connected(best_layout, problem.base_station, problem.comm_radius)

    return PSOResult(
        best_layout=best_layout,
        best_fitness=gbest_fit,
        best_coverage=gbest_cov,
        connected=connected,
        history=history,
        coverage_history=coverage_history,
        n_evaluations=n_evals,
    )


# ---------------------------------------------------------------------------
# Repair operator (Task 5)
# ---------------------------------------------------------------------------
def repair_layout(sensors: np.ndarray, problem: Problem, max_passes: int = 50):
    """Reconnect a disconnected layout by pulling stranded sensors inward.

    Strategy (greedy, deterministic): repeatedly find the connected component
    containing the base station; for every sensor NOT in that component, move
    it a step toward its nearest node that IS in the component, until it falls
    within comm range. Repeat until the whole graph is connected or a pass
    budget is exhausted. Positions are kept inside the region M.

    Returns a new (N, 2) array (the input is not modified).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    from .geometry import comm_adjacency

    layout = problem.clip_to_region(np.array(sensors, dtype=float, copy=True))
    n = layout.shape[0]
    base = np.asarray(problem.base_station, dtype=float)
    rc = problem.comm_radius
    base_idx = n  # base station is the last node

    for _ in range(max_passes):
        adj = comm_adjacency(layout, base, rc)
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        if n_comp == 1:
            break

        base_label = labels[base_idx]
        connected_mask = labels == base_label  # over all N+1 nodes
        # Node coordinates for the connected component.
        nodes = np.vstack([layout, base[None, :]])
        connected_nodes = nodes[connected_mask]

        moved = False
        for i in range(n):
            if connected_mask[i]:
                continue
            # Nearest connected node to stranded sensor i.
            d = np.linalg.norm(connected_nodes - layout[i], axis=1)
            target = connected_nodes[int(np.argmin(d))]
            dist = float(np.min(d))
            if dist <= rc:
                continue  # already in range (will join next adjacency pass)
            # Step the sensor along the line toward the target so it lands
            # just inside comm range (small margin for numerical safety).
            direction = (target - layout[i]) / (dist + 1e-12)
            step = dist - 0.95 * rc
            layout[i] = layout[i] + direction * step
            moved = True
        layout = problem.clip_to_region(layout)
        if not moved:
            break

    return layout
