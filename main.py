"""End-to-end pipeline: run PSO vs SQP, produce maps, curves, and a table.

Usage
-----
    python main.py                 # default config
    python main.py --starts 10     # more random restarts
    python main.py --sensors 10 --grid 50 --out outputs

Artifacts written to the output directory:
    comparison_table.png        summary metrics for all methods
    best_convergence.png        best-run convergence curves (PSO vs SQP)
    seed_spread.png             coverage spread across random starts
    nsensors_sweep.png          mean coverage vs number of sensors
    layout_<method>.png         best layout + coverage map + comm edges
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wsn.config import DEFAULT_CONFIG, Config
from wsn.experiment import (
    format_summary_table,
    run_experiment,
    sweep_initial_conditions,
    sweep_parameter,
)
from wsn.problem import build_problem
from wsn.visualize import (
    plot_best_convergence,
    plot_comparison_table,
    plot_coverage_map,
    plot_parameter_sweep,
    plot_seed_spread,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="WSN coverage: PSO vs SQP comparison")
    p.add_argument("--sensors", type=int, default=DEFAULT_CONFIG.n_sensors)
    p.add_argument("--grid", type=int, default=DEFAULT_CONFIG.grid_resolution)
    p.add_argument("--starts", type=int, default=DEFAULT_CONFIG.n_starts)
    p.add_argument("--rs", type=float, default=DEFAULT_CONFIG.sensing_radius)
    p.add_argument("--rc", type=float, default=DEFAULT_CONFIG.comm_radius)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    p.add_argument("--out", type=str, default=DEFAULT_CONFIG.output_dir)
    p.add_argument(
        "--quick",
        action="store_true",
        help="smaller/faster settings for a fast end-to-end smoke run",
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="skip the parameter sweep (the slowest stage)",
    )
    a = p.parse_args(argv)

    cfg = DEFAULT_CONFIG.with_overrides(
        n_sensors=a.sensors,
        grid_resolution=a.grid,
        n_starts=a.starts,
        sensing_radius=a.rs,
        comm_radius=a.rc,
        seed=a.seed,
        output_dir=a.out,
    )
    if a.quick:
        cfg = cfg.with_overrides(
            grid_resolution=min(cfg.grid_resolution, 20),
            pso_swarm_size=15,
            pso_iterations=25,
            beta_schedule=(2.0, 8.0, 25.0),
            sqp_max_iter=60,
            n_starts=min(cfg.n_starts, 4),
        )
    cfg.validate()
    return cfg, a.skip_sweep


def main(argv=None) -> None:
    cfg, skip_sweep = parse_args(argv)
    os.makedirs(cfg.output_dir, exist_ok=True)
    problem = build_problem(cfg)

    print("=" * 72)
    print("WSN Area Coverage under Connectivity  —  PSO vs SQP")
    print("=" * 72)
    print(
        f"N={cfg.n_sensors}  |P|={cfg.n_points}  rs={cfg.sensing_radius}  "
        f"rc={cfg.comm_radius}  starts={cfg.n_starts}  seed={cfg.seed}"
    )
    print("-" * 72)

    # 1) Multi-start comparison -------------------------------------------
    t0 = time.perf_counter()
    summaries = run_experiment(cfg)
    print(format_summary_table(summaries))
    print("-" * 72)

    table_path = os.path.join(cfg.output_dir, "comparison_table.png")
    conv_path = os.path.join(cfg.output_dir, "best_convergence.png")
    plot_comparison_table(summaries, save_path=table_path)
    plot_best_convergence(summaries, save_path=conv_path)

    # 2) Best layout per method -------------------------------------------
    layout_paths = []
    for method, s in summaries.items():
        br = s.best_record
        fname = f"layout_{method.replace('-', '_')}.png"
        path = os.path.join(cfg.output_dir, fname)
        plot_coverage_map(
            problem,
            br.layout,
            title=f"{method} best (C={br.coverage:.3f}, connected={br.connected})",
            show_edges=True,
            save_path=path,
        )
        layout_paths.append(path)

    # 3) Sensitivity to initial conditions --------------------------------
    spread = sweep_initial_conditions(cfg)
    spread_path = os.path.join(cfg.output_dir, "seed_spread.png")
    plot_seed_spread(spread, save_path=spread_path)

    # 4) Parameter sweep (number of sensors) ------------------------------
    sweep_path = None
    if not skip_sweep:
        sweep_cfg = cfg.with_overrides(n_starts=max(3, cfg.n_starts // 2))
        n_values = sorted({max(3, cfg.n_sensors - 2), cfg.n_sensors, cfg.n_sensors + 2})
        points = sweep_parameter("n_sensors", n_values, base_config=sweep_cfg)
        sweep_path = os.path.join(cfg.output_dir, "nsensors_sweep.png")
        plot_parameter_sweep(
            points,
            metric="mean_coverage",
            title="Mean coverage vs number of sensors",
            save_path=sweep_path,
        )

    elapsed = time.perf_counter() - t0
    print("Artifacts written:")
    for pth in [table_path, conv_path, *layout_paths, spread_path]:
        if pth:
            print(f"  {pth}")
    if sweep_path:
        print(f"  {sweep_path}")
    print(f"Total pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
