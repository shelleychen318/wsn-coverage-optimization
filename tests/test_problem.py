"""Tests for Task 1: config loading and problem/sampled-point construction."""

from __future__ import annotations

import numpy as np
import pytest

from wsn.config import Config, DEFAULT_CONFIG
from wsn.problem import build_problem


def test_config_loads_and_validates():
    cfg = DEFAULT_CONFIG
    cfg.validate()  # should not raise
    assert cfg.n_sensors >= 1
    assert cfg.n_points == cfg.grid_resolution**2


def test_config_overrides_are_independent():
    base = DEFAULT_CONFIG
    modified = base.with_overrides(n_sensors=99)
    assert modified.n_sensors == 99
    assert base.n_sensors != 99  # base unchanged (frozen dataclass)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        Config(region=(0, 0, 0, 10)).validate()  # zero width
    with pytest.raises(ValueError):
        Config(n_sensors=0).validate()
    with pytest.raises(ValueError):
        Config(base_station=(999, 999)).validate()  # outside region


def test_sampled_points_count_matches_resolution():
    cfg = DEFAULT_CONFIG.with_overrides(grid_resolution=25)
    problem = build_problem(cfg)
    assert len(problem.points) == 25 * 25


def test_sampled_points_lie_inside_region():
    cfg = DEFAULT_CONFIG
    problem = build_problem(cfg)
    x_min, y_min, x_max, y_max = cfg.region
    pts = problem.points
    assert np.all(pts[:, 0] >= x_min - 1e-9)
    assert np.all(pts[:, 0] <= x_max + 1e-9)
    assert np.all(pts[:, 1] >= y_min - 1e-9)
    assert np.all(pts[:, 1] <= y_max + 1e-9)


def test_random_layout_inside_region_and_reproducible():
    cfg = DEFAULT_CONFIG
    problem = build_problem(cfg)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    layout1 = problem.random_layout(rng1)
    layout2 = problem.random_layout(rng2)
    assert layout1.shape == (cfg.n_sensors, 2)
    assert np.allclose(layout1, layout2)  # same seed -> same layout
    x_min, y_min, x_max, y_max = cfg.region
    assert np.all(layout1[:, 0] >= x_min) and np.all(layout1[:, 0] <= x_max)
    assert np.all(layout1[:, 1] >= y_min) and np.all(layout1[:, 1] <= y_max)


def test_clip_to_region():
    cfg = DEFAULT_CONFIG
    problem = build_problem(cfg)
    outside = np.array([[-5.0, -5.0], [100.0, 100.0]])
    clipped = problem.clip_to_region(outside)
    x_min, y_min, x_max, y_max = cfg.region
    assert np.allclose(clipped[0], [x_min, y_min])
    assert np.allclose(clipped[1], [x_max, y_max])
