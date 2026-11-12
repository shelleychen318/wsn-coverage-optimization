"""Task 2 demo: compute coverage for a hand-placed layout and render a map."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import coverage_ratio
from wsn.problem import build_problem
from wsn.visualize import plot_coverage_map


def main() -> None:
    config = DEFAULT_CONFIG
    problem = build_problem(config)

    # A hand-placed layout: a loose ring of sensors around the base station.
    angles = np.linspace(0, 2 * np.pi, config.n_sensors, endpoint=False)
    radius = 3.0
    cx, cy = problem.base_station
    sensors = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
    sensors = problem.clip_to_region(sensors)

    ratio = coverage_ratio(sensors, problem.points, config.sensing_radius)
    print(f"Hand-placed ring layout coverage ratio C(S) = {ratio:.4f}")

    out = os.path.join(config.output_dir, "task2_coverage_map.png")
    plot_coverage_map(problem, sensors, title=f"Hand-placed layout (C={ratio:.3f})", save_path=out)
    print(f"Coverage map saved to: {out}")


if __name__ == "__main__":
    main()
