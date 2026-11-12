"""Tests for Task 5: PSO repair operator and constraint-mode dispatch."""

from __future__ import annotations

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import is_connected
from wsn.problem import build_problem
from wsn.pso import repair_layout, run_pso, solve_pso


def _fast_config(**overrides):
    base = DEFAULT_CONFIG.with_overrides(
        n_sensors=6, grid_resolution=25, pso_swarm_size=20, pso_iterations=30
    )
    return base.with_overrides(**overrides)


def test_repair_makes_disconnected_layout_connected():
    cfg = _fast_config()
    problem = build_problem(cfg)
    disconnected = np.array(
        [[0.1, 0.1], [9.9, 9.9], [0.1, 9.9], [9.9, 0.1], [5.1, 5.0], [5.0, 5.1]]
    )
    assert not is_connected(disconnected, problem.base_station, cfg.comm_radius)
    repaired = repair_layout(disconnected, problem)
    assert is_connected(repaired, problem.base_station, cfg.comm_radius)


def test_repair_keeps_points_inside_region():
    cfg = _fast_config()
    problem = build_problem(cfg)
    disconnected = np.array(
        [[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0], [5.0, 5.0], [5.5, 5.5]]
    )
    repaired = repair_layout(disconnected, problem)
    x_min, y_min, x_max, y_max = cfg.region
    assert np.all(repaired[:, 0] >= x_min - 1e-9)
    assert np.all(repaired[:, 0] <= x_max + 1e-9)
    assert np.all(repaired[:, 1] >= y_min - 1e-9)
    assert np.all(repaired[:, 1] <= y_max + 1e-9)


def test_repair_leaves_connected_layout_connected():
    cfg = _fast_config()
    problem = build_problem(cfg)
    connected = np.array(
        [[5.0, 5.5], [5.5, 5.0], [4.5, 5.0], [5.0, 4.5], [6.0, 5.0], [5.0, 6.0]]
    )
    repaired = repair_layout(connected, problem)
    assert is_connected(repaired, problem.base_station, cfg.comm_radius)


def test_repair_mode_run_is_feasible():
    cfg = _fast_config(pso_constraint_mode="repair", comm_radius=3.0)
    problem = build_problem(cfg)
    result = solve_pso(problem, cfg, seed=3)
    # Repair mode should return a connected layout.
    assert result.connected is True
    assert result.best_coverage > 0.3


def test_solve_pso_dispatches_on_mode():
    problem_pen = build_problem(_fast_config(pso_constraint_mode="penalty"))
    problem_rep = build_problem(_fast_config(pso_constraint_mode="repair"))
    r_pen = solve_pso(problem_pen, seed=5)
    r_rep = solve_pso(problem_rep, seed=5)
    # Both produce valid results; just assert they ran and returned layouts.
    assert r_pen.best_layout.shape == (6, 2)
    assert r_rep.best_layout.shape == (6, 2)
