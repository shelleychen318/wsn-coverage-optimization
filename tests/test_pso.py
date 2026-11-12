"""Tests for Task 4: hand-written PSO with penalty constraint handling."""

from __future__ import annotations

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import coverage_ratio, is_connected
from wsn.problem import build_problem
from wsn.pso import _disconnection_penalty, run_pso


def _fast_config():
    # Small, quick instance for tests.
    return DEFAULT_CONFIG.with_overrides(
        n_sensors=6,
        grid_resolution=25,
        pso_swarm_size=25,
        pso_iterations=40,
    )


def test_pso_improves_coverage_over_iterations():
    cfg = _fast_config()
    problem = build_problem(cfg)
    result = run_pso(problem, cfg, seed=1)
    # Best fitness should be non-decreasing (gbest only ever improves).
    hist = result.history
    assert all(hist[i + 1] >= hist[i] - 1e-12 for i in range(len(hist) - 1))
    # And it should end clearly above a single-sensor baseline.
    assert result.best_coverage > 0.3


def test_pso_returns_connected_layout_on_easy_instance():
    cfg = _fast_config().with_overrides(comm_radius=4.0)
    problem = build_problem(cfg)
    result = run_pso(problem, cfg, seed=2)
    assert result.connected is True


def test_penalty_lowers_fitness_of_disconnected_layout():
    cfg = _fast_config()
    problem = build_problem(cfg)
    # A disconnected layout should incur a positive penalty.
    disconnected = np.array(
        [[0.1, 0.1], [9.9, 9.9], [0.1, 9.9], [9.9, 0.1], [5.0, 5.2], [5.2, 5.0]]
    )
    assert not is_connected(disconnected, problem.base_station, cfg.comm_radius)
    pen = _disconnection_penalty(disconnected, problem, cfg.pso_penalty_weight)
    assert pen > 0.0
    # A connected cluster near the base station pays ~0.
    connected = np.array(
        [[5.0, 5.5], [5.5, 5.0], [4.5, 5.0], [5.0, 4.5], [6.0, 5.0], [5.0, 6.0]]
    )
    assert is_connected(connected, problem.base_station, cfg.comm_radius)
    pen_ok = _disconnection_penalty(connected, problem, cfg.pso_penalty_weight)
    assert pen_ok < pen


def test_pso_is_reproducible_with_seed():
    cfg = _fast_config()
    problem = build_problem(cfg)
    r1 = run_pso(problem, cfg, seed=7)
    r2 = run_pso(problem, cfg, seed=7)
    assert np.isclose(r1.best_fitness, r2.best_fitness)
    assert np.allclose(r1.best_layout, r2.best_layout)
