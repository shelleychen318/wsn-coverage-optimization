"""Tests for Task 8: multi-start experiment runner and metric collection."""

from __future__ import annotations

from wsn.config import DEFAULT_CONFIG
from wsn.experiment import format_summary_table, run_experiment


def _fast_config():
    return DEFAULT_CONFIG.with_overrides(
        n_sensors=5,
        grid_resolution=20,
        pso_swarm_size=15,
        pso_iterations=20,
        beta_schedule=(2.0, 8.0, 25.0),
        sqp_max_iter=60,
        n_starts=3,
    )


def test_runner_produces_one_record_per_seed_per_method():
    cfg = _fast_config()
    summaries = run_experiment(cfg)
    assert set(summaries.keys()) == {"PSO-penalty", "PSO-repair", "SQP"}
    for method, s in summaries.items():
        assert len(s.records) == cfg.n_starts
        for r in s.records:
            assert r.method == method


def test_metrics_populated_and_in_range():
    cfg = _fast_config()
    summaries = run_experiment(cfg)
    for s in summaries.values():
        assert 0.0 <= s.worst_coverage <= s.mean_coverage <= s.best_coverage <= 1.0
        assert 0.0 <= s.feasibility_rate <= 1.0
        assert s.mean_time >= 0.0
        assert s.mean_evaluations > 0
        for r in s.records:
            # Convergence history recorded.
            assert len(r.convergence) >= 1
            # Layout present and shaped correctly.
            assert r.layout.shape == (cfg.n_sensors, 2)


def test_connectivity_validated_with_true_check():
    # The recorded 'connected' flag must come from the true BFS check: a run
    # flagged connected should pass is_connected on its own layout.
    from wsn.geometry import is_connected

    cfg = _fast_config()
    summaries = run_experiment(cfg)
    problem_base = summaries["SQP"].records[0].layout
    assert problem_base is not None
    for s in summaries.values():
        for r in s.records:
            true_conn = is_connected(
                r.layout, DEFAULT_CONFIG.base_station, cfg.comm_radius
            )
            # Note: comm_radius default matches cfg (not overridden here).
            assert r.connected == true_conn


def test_summary_table_contains_all_methods():
    cfg = _fast_config()
    summaries = run_experiment(cfg)
    table = format_summary_table(summaries)
    for method in ("PSO-penalty", "PSO-repair", "SQP"):
        assert method in table
