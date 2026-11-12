"""Tests for Task 9: sensitivity sweeps."""

from __future__ import annotations

from wsn.config import DEFAULT_CONFIG
from wsn.experiment import sweep_initial_conditions, sweep_parameter


def _fast_config():
    return DEFAULT_CONFIG.with_overrides(
        n_sensors=5,
        grid_resolution=18,
        pso_swarm_size=12,
        pso_iterations=15,
        beta_schedule=(2.0, 8.0),
        sqp_max_iter=40,
        n_starts=3,
    )


def test_sweep_initial_conditions_captures_per_seed_values():
    cfg = _fast_config()
    spread = sweep_initial_conditions(cfg)
    assert set(spread.keys()) == {"PSO-penalty", "PSO-repair", "SQP"}
    for method, values in spread.items():
        assert len(values) == cfg.n_starts  # one outcome per random start


def test_sweep_parameter_produces_grid():
    cfg = _fast_config()
    values = [4, 6]
    points = sweep_parameter("n_sensors", values, base_config=cfg)
    # 3 methods x 2 parameter values.
    assert len(points) == 3 * len(values)
    seen = {(p.method, p.param_value) for p in points}
    assert len(seen) == 3 * len(values)
    for p in points:
        assert p.param_name == "n_sensors"
        assert len(p.coverages) == cfg.n_starts
        assert 0.0 <= p.feasibility_rate <= 1.0


def test_sweep_variance_captured_per_method():
    cfg = _fast_config()
    points = sweep_parameter("comm_radius", [3.0, 4.0], base_config=cfg)
    for p in points:
        # std is a non-negative float.
        assert p.std_coverage >= 0.0
