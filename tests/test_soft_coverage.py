"""Tests for Task 6: smooth coverage objective and continuation behavior."""

from __future__ import annotations

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import coverage_ratio, soft_coverage_ratio
from wsn.problem import build_problem


def _layout():
    return np.array([[3.0, 3.0], [7.0, 7.0], [3.0, 7.0], [7.0, 3.0], [5.0, 5.0]])


def test_soft_coverage_in_unit_interval():
    problem = build_problem(DEFAULT_CONFIG.with_overrides(grid_resolution=30))
    s = soft_coverage_ratio(_layout(), problem.points, rs=1.8, beta=5.0)
    assert 0.0 <= s <= 1.0


def test_soft_approaches_hard_as_beta_grows():
    problem = build_problem(DEFAULT_CONFIG.with_overrides(grid_resolution=60))
    sensors = _layout()
    hard = coverage_ratio(sensors, problem.points, rs=1.8)
    diffs = []
    for beta in (1.0, 5.0, 20.0, 80.0, 200.0):
        soft = soft_coverage_ratio(sensors, problem.points, rs=1.8, beta=beta)
        diffs.append(abs(soft - hard))
    # Error to the Boolean coverage should shrink as beta increases.
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 0.02


def test_soft_coverage_has_nonzero_gradient_where_hard_is_flat():
    # Move a single sensor slightly; the HARD metric may not change (flat),
    # but the SOFT metric must respond (non-zero finite-difference gradient).
    problem = build_problem(DEFAULT_CONFIG.with_overrides(grid_resolution=40))
    # Off-center so the x-perturbation is not cancelled by grid symmetry.
    base = np.array([[3.3, 4.1]])
    rs, beta = 1.8, 5.0

    def soft_at(dx):
        s = base.copy()
        s[0, 0] += dx
        return soft_coverage_ratio(s, problem.points, rs, beta)

    h = 1e-4
    grad = (soft_at(h) - soft_at(-h)) / (2 * h)
    assert abs(grad) > 1e-6  # smooth objective is responsive


def test_soft_coverage_finite_and_stable_for_large_beta():
    problem = build_problem(DEFAULT_CONFIG.with_overrides(grid_resolution=30))
    s = soft_coverage_ratio(_layout(), problem.points, rs=1.8, beta=500.0)
    assert np.isfinite(s)
