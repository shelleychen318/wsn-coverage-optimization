"""Task 1 demo: print the problem setup and confirm the environment is wired up."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.problem import build_problem


def main() -> None:
    config = DEFAULT_CONFIG
    problem = build_problem(config)

    print("=" * 60)
    print("WSN Coverage Problem Setup")
    print("=" * 60)
    print(f"Sensors to place (N):      {config.n_sensors}")
    print(f"Region M (xmin,ymin,xmax,ymax): {config.region}")
    print(f"Region size:               {config.region_width} x {config.region_height}")
    print(f"Base station:              {tuple(problem.base_station)}")
    print(f"Sensing radius r_s:        {config.sensing_radius}")
    print(f"Comm radius r_c:           {config.comm_radius}")
    print(f"Grid resolution:           {config.grid_resolution}")
    print(f"Sampled points |P|:        {len(problem.points)}")
    print("-" * 60)

    # Confirm numeric stack is importable and functioning.
    rng = np.random.default_rng(config.seed)
    layout = problem.random_layout(rng)
    print(f"Example random layout (first 3 of {len(layout)} sensors):")
    for i, (x, y) in enumerate(layout[:3]):
        print(f"  sensor {i}: ({x:.3f}, {y:.3f})")
    print("Environment wired up: numpy OK.")


if __name__ == "__main__":
    main()
