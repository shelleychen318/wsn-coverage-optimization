"""Task 9 demo: sensitivity to initial conditions and a parameter sweep."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsn.config import DEFAULT_CONFIG
from wsn.experiment import sweep_initial_conditions, sweep_parameter
from wsn.visualize import plot_parameter_sweep, plot_seed_spread


def main() -> None:
    config = DEFAULT_CONFIG.with_overrides(n_starts=8)

    # --- Sensitivity to initial conditions ---------------------------------
    print("=" * 60)
    print("Sensitivity to initial conditions (coverage per seed)")
    print("=" * 60)
    spread = sweep_initial_conditions(config)
    for method, values in spread.items():
        lo, hi = min(values), max(values)
        print(f"{method:<14} range=[{lo:.3f}, {hi:.3f}]  spread={hi - lo:.3f}")

    os.makedirs(config.output_dir, exist_ok=True)
    spread_path = os.path.join(config.output_dir, "task9_seed_spread.png")
    plot_seed_spread(spread, save_path=spread_path)
    print(f"Saved: {spread_path}")

    # --- Parameter sweep: number of sensors --------------------------------
    print("-" * 60)
    print("Parameter sweep: n_sensors in {6, 8, 10}")
    sweep_cfg = config.with_overrides(n_starts=4)
    points = sweep_parameter("n_sensors", [6, 8, 10], base_config=sweep_cfg)
    for p in points:
        print(
            f"  N={int(p.param_value):<3} {p.method:<14} "
            f"mean={p.mean_coverage:.3f} feas%={100 * p.feasibility_rate:.0f}"
        )
    sweep_path = os.path.join(config.output_dir, "task9_nsensors_sweep.png")
    plot_parameter_sweep(points, metric="mean_coverage",
                         title="Mean coverage vs number of sensors",
                         save_path=sweep_path)
    print(f"Saved: {sweep_path}")


if __name__ == "__main__":
    main()
