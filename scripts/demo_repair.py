"""Task 5 demo: compare penalty-PSO vs repair-PSO on the same seed."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsn.config import DEFAULT_CONFIG
from wsn.problem import build_problem
from wsn.pso import solve_pso
from wsn.visualize import plot_convergence, plot_coverage_map


def main() -> None:
    seed = DEFAULT_CONFIG.seed

    cfg_pen = DEFAULT_CONFIG.with_overrides(pso_constraint_mode="penalty")
    cfg_rep = DEFAULT_CONFIG.with_overrides(pso_constraint_mode="repair")
    problem_pen = build_problem(cfg_pen)
    problem_rep = build_problem(cfg_rep)

    res_pen = solve_pso(problem_pen, cfg_pen, seed=seed)
    res_rep = solve_pso(problem_rep, cfg_rep, seed=seed)

    print("=" * 60)
    print(f"PSO constraint-handling comparison (seed={seed})")
    print("=" * 60)
    print(f"{'mode':<10}{'coverage':>12}{'connected':>12}{'evals':>10}")
    print("-" * 44)
    print(f"{'penalty':<10}{res_pen.best_coverage:>12.4f}{str(res_pen.connected):>12}{res_pen.n_evaluations:>10}")
    print(f"{'repair':<10}{res_rep.best_coverage:>12.4f}{str(res_rep.connected):>12}{res_rep.n_evaluations:>10}")

    os.makedirs(DEFAULT_CONFIG.output_dir, exist_ok=True)
    curve_path = os.path.join(DEFAULT_CONFIG.output_dir, "task5_penalty_vs_repair.png")
    plot_convergence(
        {
            "penalty": res_pen.coverage_history,
            "repair": res_rep.coverage_history,
        },
        title="PSO: penalty vs repair convergence",
        save_path=curve_path,
    )
    pen_map = os.path.join(DEFAULT_CONFIG.output_dir, "task5_penalty_layout.png")
    rep_map = os.path.join(DEFAULT_CONFIG.output_dir, "task5_repair_layout.png")
    plot_coverage_map(problem_pen, res_pen.best_layout,
                      title=f"Penalty PSO (C={res_pen.best_coverage:.3f})",
                      show_edges=True, save_path=pen_map)
    plot_coverage_map(problem_rep, res_rep.best_layout,
                      title=f"Repair PSO (C={res_rep.best_coverage:.3f})",
                      show_edges=True, save_path=rep_map)
    print(f"Saved: {curve_path}")
    print(f"Saved: {pen_map}")
    print(f"Saved: {rep_map}")


if __name__ == "__main__":
    main()
