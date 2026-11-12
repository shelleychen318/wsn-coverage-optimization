"""Sequential Quadratic Programming (SLSQP) solver with continuation.

We maximize the smooth coverage objective (minimize its negative) subject to:
  * box constraints keeping every sensor inside the rectangular region M, and
  * a nonlinear inequality lambda_2(S) >= eps on the smooth algebraic
    connectivity (Fiedler value), which keeps the communication graph
    connected with a differentiable signal.

Because both the objective and the connectivity surrogate depend on a sharpness
parameter beta, we wrap SLSQP in a continuation (homotopy) loop: solve at a
small beta, then warm-start the next solve at a larger beta. This gradually
sharpens the smooth problem toward the true Boolean model while keeping each
sub-problem well-conditioned. Gradients are supplied by SLSQP's finite
differences (per the project decision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from .config import Config
from .geometry import (
    algebraic_connectivity,
    coverage_ratio,
    is_connected,
    soft_coverage_ratio,
)
from .problem import Problem


@dataclass
class SQPResult:
    """Outcome of a single (multi-stage) SQP run."""

    best_layout: np.ndarray  # (N, 2)
    best_coverage: float  # TRUE (hard) coverage of the final layout
    connected: bool  # TRUE connectivity of the final layout
    history: List[float] = field(default_factory=list)  # true coverage per stage
    soft_history: List[float] = field(default_factory=list)  # soft coverage per stage
    beta_history: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    success: bool = False


def run_sqp(
    problem: Problem,
    config: Optional[Config] = None,
    seed: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
) -> SQPResult:
    """Run SLSQP with a beta-continuation schedule and return the best layout."""
    config = config or problem.config
    seed = config.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    n = problem.n_sensors
    rs = problem.sensing_radius
    rc = problem.comm_radius
    rc_eff = max(1e-3, rc - config.sqp_comm_margin)  # margin for true feasibility
    base = problem.base_station
    eps = config.lambda2_epsilon
    points = problem.points

    # Starting layout (flattened). Warm-started across continuation stages.
    if x0 is None:
        x0 = problem.random_layout(rng).reshape(-1)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(-1)

    bounds = problem.bounds  # list of (min, max) per variable
    n_evals = 0

    history: List[float] = []
    soft_history: List[float] = []
    beta_history: List[float] = []

    current = x0.copy()
    last_success = False

    for beta in config.beta_schedule:
        def neg_soft_coverage(vec: np.ndarray, beta=beta) -> float:
            nonlocal n_evals
            n_evals += 1
            layout = vec.reshape(n, 2)
            return -soft_coverage_ratio(layout, points, rs, beta)

        def connectivity_constraint(vec: np.ndarray, beta=beta) -> float:
            # Must be >= 0 for SLSQP 'ineq': lambda_2 - eps >= 0.
            # Use an effective comm radius (rc - margin): SLSQP settles sensors
            # on the active boundary, so the margin keeps the TRUE edges
            # (distance <= rc) inside range and makes the smooth constraint a
            # sufficient condition for true connectivity.
            layout = vec.reshape(n, 2)
            return algebraic_connectivity(layout, base, rc_eff, beta) - eps

        constraints = [{"type": "ineq", "fun": connectivity_constraint}]

        res = minimize(
            neg_soft_coverage,
            current,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": config.sqp_max_iter, "ftol": 1e-7},
        )

        current = res.x  # warm-start the next (sharper) stage
        last_success = bool(res.success)

        layout = current.reshape(n, 2)
        true_cov = coverage_ratio(layout, points, rs)
        soft_cov = soft_coverage_ratio(layout, points, rs, beta)
        history.append(true_cov)
        soft_history.append(soft_cov)
        beta_history.append(float(beta))

    best_layout = current.reshape(n, 2)
    best_cov = coverage_ratio(best_layout, points, rs)
    connected = is_connected(best_layout, base, rc)

    return SQPResult(
        best_layout=best_layout,
        best_coverage=best_cov,
        connected=connected,
        history=history,
        soft_history=soft_history,
        beta_history=beta_history,
        n_evaluations=n_evals,
        success=last_success,
    )
