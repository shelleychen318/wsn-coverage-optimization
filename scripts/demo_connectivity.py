"""Task 3 demo: compare a connected and a disconnected layout.

Prints connectivity status + smooth lambda_2 for each, and draws the
communication edges on the coverage map.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wsn.config import DEFAULT_CONFIG
from wsn.geometry import algebraic_connectivity, coverage_ratio, is_connected
from wsn.problem import build_problem
from wsn.visualize import plot_coverage_map


def report(name, problem, sensors, beta):
    base, rc, rs = problem.base_station, problem.comm_radius, problem.sensing_radius
    connected = is_connected(sensors, base, rc)
    lam2 = algebraic_connectivity(sensors, base, rc, beta=beta)
    cov = coverage_ratio(sensors, problem.points, rs)
    print(f"[{name}] connected={connected}  lambda2={lam2:.4f}  coverage={cov:.3f}")
    return connected, lam2


def main() -> None:
    config = DEFAULT_CONFIG
    problem = build_problem(config)
    beta = config.beta_schedule[-1]

    # Connected: sensors chained outward from the base station.
    connected_layout = np.array(
        [
            [5.0, 6.5], [5.0, 8.0], [6.5, 5.0], [8.0, 5.0],
            [3.5, 5.0], [2.0, 5.0], [5.0, 3.5], [5.0, 2.0],
        ]
    )

    # Disconnected: two sensors stranded in far corners.
    disconnected_layout = np.array(
        [
            [5.0, 6.5], [6.5, 5.0], [3.5, 5.0], [5.0, 3.5],
            [0.2, 0.2], [9.8, 9.8], [0.2, 9.8], [9.8, 0.2],
        ]
    )

    print("=" * 60)
    print("Connectivity comparison (beta = %.1f)" % beta)
    print("=" * 60)
    report("connected   ", problem, connected_layout, beta)
    report("disconnected", problem, disconnected_layout, beta)

    os.makedirs(config.output_dir, exist_ok=True)
    out_c = os.path.join(config.output_dir, "task3_connected.png")
    out_d = os.path.join(config.output_dir, "task3_disconnected.png")
    plot_coverage_map(problem, connected_layout, title="Connected layout",
                      show_edges=True, save_path=out_c)
    plot_coverage_map(problem, disconnected_layout, title="Disconnected layout",
                      show_edges=True, save_path=out_d)
    print(f"Saved: {out_c}")
    print(f"Saved: {out_d}")


if __name__ == "__main__":
    main()
