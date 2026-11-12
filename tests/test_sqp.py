"""Tests for Task 7: SLSQP solver with continuation and connectivity."""

from __future__ import annotations

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import coverage_ratio, is_connected
from wsn.problem import build_problem
from wsn.sqp import run_sqp


def _fast_config(**overrides):
    base = DEFAULT_CONFIG.with_overrides(
        n_sensors=5,
        grid_resolution=25,
        comm_radius=4.0,
        beta_schedule=(2.0, 8.0, 25.0),
        sqp_max_iter=80,
    )
    return base.with_overrides(**overrides)


def test_sqp_improves_over_start():
    cfg = _fast_config()
    problem = build_problem(cfg)
    rng = np.random.default_rng(10)
    x0 = problem.random_layout(rng)
    start_cov = coverage_ratio(x0, problem.points, cfg.sensing_radius)
    result = run_sqp(problem, cfg, x0=x0)
    assert result.best_coverage >= start_cov - 1e-6


def test_sqp_returns_connected_layout_on_easy_instance():
    cfg = _fast_config(comm_radius=5.0)
    problem = build_problem(cfg)
    result = run_sqp(problem, cfg, seed=4)
    assert result.connected is True
    # Verified with the TRUE BFS check, not the smooth surrogate.
    assert is_connected(result.best_layout, problem.base_station, cfg.comm_radius)


def test_sqp_records_continuation_history():
    cfg = _fast_config()
    problem = build_problem(cfg)
    result = run_sqp(problem, cfg, seed=1)
    assert len(result.history) == len(cfg.beta_schedule)
    assert len(result.beta_history) == len(cfg.beta_schedule)
    assert result.beta_history == list(cfg.beta_schedule)


def test_sqp_layout_inside_region():
    cfg = _fast_config()
    problem = build_problem(cfg)
    result = run_sqp(problem, cfg, seed=2)
    x_min, y_min, x_max, y_max = cfg.region
    pts = result.best_layout
    assert np.all(pts[:, 0] >= x_min - 1e-6)
    assert np.all(pts[:, 0] <= x_max + 1e-6)
    assert np.all(pts[:, 1] >= y_min - 1e-6)
    assert np.all(pts[:, 1] <= y_max + 1e-6)


def test_sqp_reproducible_with_seed():
    cfg = _fast_config()
    problem = build_problem(cfg)
    r1 = run_sqp(problem, cfg, seed=9)
    r2 = run_sqp(problem, cfg, seed=9)
    assert np.allclose(r1.best_layout, r2.best_layout)
