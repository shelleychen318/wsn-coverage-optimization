"""Task 7 demo: run SQP from one seed, report, plot curve + layout."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsn.config import DEFAULT_CONFIG
from wsn.problem import build_problem
from wsn.sqp import run_sqp
from wsn.visualize import plot_convergence, plot_coverage_map


def main() -> None:
    config = DEFAULT_CONFIG
    problem = build_problem(config)

    result = run_sqp(problem, config, seed=config.seed)

    print("=" * 60)
    print("SQP (SLSQP + continuation) result")
    print("=" * 60)
    print(f"Final TRUE coverage:  {result.best_coverage:.4f}")
    print(f"Connected:            {result.connected}")
    print(f"SLSQP success flag:   {result.success}")
    print(f"Evaluations:          {result.n_evaluations}")
    print("-" * 60)
    print("Continuation stages (beta -> true coverage / soft coverage):")
    for b, h, sh in zip(result.beta_history, result.history, result.soft_history):
        print(f"  beta={b:7.1f}   true={h:.4f}   soft={sh:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    curve_path = os.path.join(config.output_dir, "task7_sqp_convergence.png")
    map_path = os.path.join(config.output_dir, "task7_sqp_layout.png")
    plot_convergence(
        {"SQP true coverage": result.history, "SQP soft coverage": result.soft_history},
        title="SQP convergence across continuation stages",
        ylabel="coverage ratio",
        save_path=curve_path,
    )
    plot_coverage_map(
        problem,
        result.best_layout,
        title=f"SQP layout (C={result.best_coverage:.3f}, connected={result.connected})",
        show_edges=True,
        save_path=map_path,
    )
    print(f"Saved: {curve_path}")
    print(f"Saved: {map_path}")


if __name__ == "__main__":
    main()
