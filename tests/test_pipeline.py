"""Tests for Task 10: visualization outputs and the end-to-end main pipeline."""

from __future__ import annotations

import os

from wsn.config import DEFAULT_CONFIG
from wsn.experiment import run_experiment
from wsn.visualize import (
    plot_best_convergence,
    plot_comparison_table,
    plot_seed_spread,
)

import main as pipeline


def _fast_config(tmp_path):
    return DEFAULT_CONFIG.with_overrides(
        n_sensors=5,
        grid_resolution=18,
        pso_swarm_size=12,
        pso_iterations=15,
        beta_schedule=(2.0, 8.0),
        sqp_max_iter=40,
        n_starts=3,
        output_dir=str(tmp_path),
    )


def test_comparison_table_and_curves_saved(tmp_path):
    cfg = _fast_config(tmp_path)
    summaries = run_experiment(cfg)

    table_path = os.path.join(cfg.output_dir, "comparison_table.png")
    conv_path = os.path.join(cfg.output_dir, "best_convergence.png")
    spread_path = os.path.join(cfg.output_dir, "seed_spread.png")

    plot_comparison_table(summaries, save_path=table_path)
    plot_best_convergence(summaries, save_path=conv_path)
    plot_seed_spread(
        {m: [r.coverage for r in s.records] for m, s in summaries.items()},
        save_path=spread_path,
    )

    assert os.path.exists(table_path)
    assert os.path.exists(conv_path)
    assert os.path.exists(spread_path)


def test_main_pipeline_runs_end_to_end(tmp_path):
    # Drive the CLI with fast flags; it should run with no manual steps and
    # produce the core artifacts.
    argv = [
        "--quick",
        "--sensors", "5",
        "--grid", "18",
        "--starts", "3",
        "--out", str(tmp_path),
        "--skip-sweep",
    ]
    pipeline.main(argv)

    expected = [
        "comparison_table.png",
        "best_convergence.png",
        "seed_spread.png",
    ]
    for name in expected:
        assert os.path.exists(os.path.join(str(tmp_path), name)), name
    # At least one per-method layout map exists.
    layouts = [f for f in os.listdir(str(tmp_path)) if f.startswith("layout_")]
    assert len(layouts) >= 1
