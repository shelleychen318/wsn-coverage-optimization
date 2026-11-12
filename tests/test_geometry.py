"""Tests for Task 2: vectorized distances and the true (hard) coverage metric."""

from __future__ import annotations

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import coverage_mask, coverage_ratio, pairwise_distances
from wsn.problem import build_problem


def test_pairwise_distances_shape_and_values():
    a = np.array([[0.0, 0.0], [3.0, 0.0]])
    b = np.array([[0.0, 4.0]])
    d = pairwise_distances(a, b)
    assert d.shape == (2, 1)
    assert np.isclose(d[0, 0], 4.0)
    assert np.isclose(d[1, 0], 5.0)  # 3-4-5 triangle


def test_single_sensor_covers_points_within_rs():
    # Grid over a 10x10 region, one sensor at the center.
    cfg = DEFAULT_CONFIG.with_overrides(grid_resolution=101)
    problem = build_problem(cfg)
    rs = 2.0
    sensor = np.array([[5.0, 5.0]])
    mask = coverage_mask(sensor, problem.points, rs)

    # Exactly the points whose distance to center <= rs should be covered.
    dist = np.linalg.norm(problem.points - np.array([5.0, 5.0]), axis=1)
    expected = dist <= rs
    assert np.array_equal(mask, expected)


def test_overlap_not_double_counted():
    # Two sensors at the SAME location cover the same set as one sensor.
    cfg = DEFAULT_CONFIG.with_overrides(grid_resolution=50)
    problem = build_problem(cfg)
    rs = 2.0
    one = np.array([[5.0, 5.0]])
    two = np.array([[5.0, 5.0], [5.0, 5.0]])
    assert np.isclose(
        coverage_ratio(one, problem.points, rs),
        coverage_ratio(two, problem.points, rs),
    )
    # Ratio is a fraction in [0, 1], never exceeding 1 despite overlap.
    assert 0.0 <= coverage_ratio(two, problem.points, rs) <= 1.0


def test_full_grid_layout_approaches_full_coverage():
    # Dense lattice of sensors with generous radius should cover ~everything.
    cfg = DEFAULT_CONFIG.with_overrides(grid_resolution=40)
    problem = build_problem(cfg)
    xs = np.linspace(0.5, 9.5, 6)
    ys = np.linspace(0.5, 9.5, 6)
    gx, gy = np.meshgrid(xs, ys)
    sensors = np.column_stack([gx.ravel(), gy.ravel()])
    ratio = coverage_ratio(sensors, problem.points, rs=2.0)
    assert ratio > 0.98


def test_coverage_ratio_bounds():
    cfg = DEFAULT_CONFIG
    problem = build_problem(cfg)
    # A single far-away sensor covers almost nothing but ratio stays >= 0.
    sensor = np.array([[0.0, 0.0]])
    r = coverage_ratio(sensor, problem.points, rs=0.1)
    assert 0.0 <= r <= 1.0
