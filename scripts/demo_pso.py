"""Task 4 demo: run PSO from one seed, report results, plot curve + map."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsn.config import DEFAULT_CONFIG
from wsn.problem import build_problem
from wsn.pso import run_pso
from wsn.visualize import plot_convergence, plot_coverage_map


def main() -> None:
    config = DEFAULT_CONFIG
    problem = build_problem(config)

    result = run_pso(problem, config, seed=config.seed)

    print("=" * 60)
    print("PSO (penalty mode) result")
    print("=" * 60)
    print(f"Best coverage ratio:  {result.best_coverage:.4f}")
    print(f"Best fitness:         {result.best_fitness:.4f}")
    print(f"Connected:            {result.connected}")
    print(f"Evaluations:          {result.n_evaluations}")

    os.makedirs(config.output_dir, exist_ok=True)
    curve_path = os.path.join(config.output_dir, "task4_pso_convergence.png")
    map_path = os.path.join(config.output_dir, "task4_pso_layout.png")
    plot_convergence(
        {"PSO coverage": result.coverage_history},
        title="PSO convergence",
        save_path=curve_path,
    )
    plot_coverage_map(
        problem,
        result.best_layout,
        title=f"PSO layout (C={result.best_coverage:.3f}, connected={result.connected})",
        show_edges=True,
        save_path=map_path,
    )
    print(f"Saved: {curve_path}")
    print(f"Saved: {map_path}")


if __name__ == "__main__":
    main()
