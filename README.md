# WSN Area Coverage Maximization under Network Connectivity

Place N wireless sensors in a rectangular region to **maximize area coverage**
while keeping the communication graph **connected** back to a fixed base
station. The problem is solved two ways and compared:

- **PSO** (Particle Swarm Optimization) — hand-written, stochastic,
  population-based, gradient-free. Connectivity handled by a **penalty** on
  disconnected layouts or by a **repair** operator.
- **SQP** (Sequential Quadratic Programming, via SciPy `SLSQP`) —
  deterministic, gradient-based, local. Uses **smooth approximations** of the
  (non-differentiable) coverage objective and the (discrete) connectivity
  constraint, sharpened over a **continuation schedule**.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the full comparison pipeline

```bash
.venv/bin/python main.py              # default problem
.venv/bin/python main.py --quick      # fast smoke run
.venv/bin/python main.py --starts 10 --sensors 10 --skip-sweep
```

Artifacts are written to `outputs/`:
- `comparison_table.png` — metrics for all methods
- `best_convergence.png` — best-run convergence (PSO vs SQP)
- `seed_spread.png` — coverage spread across random starts (sensitivity)
- `nsensors_sweep.png` — mean coverage vs number of sensors
- `layout_<method>.png` — best layout with sensing circles + comm edges

## Per-task demos

Each task has a standalone demo under `scripts/`:

```bash
.venv/bin/python scripts/demo_setup.py         # Task 1: problem setup
.venv/bin/python scripts/demo_coverage.py      # Task 2: coverage map
.venv/bin/python scripts/demo_connectivity.py  # Task 3: connectivity + lambda2
.venv/bin/python scripts/demo_pso.py           # Task 4: PSO (penalty)
.venv/bin/python scripts/demo_repair.py        # Task 5: penalty vs repair
.venv/bin/python scripts/demo_soft_coverage.py # Task 6: soft vs hard coverage
.venv/bin/python scripts/demo_sqp.py           # Task 7: SQP + continuation
.venv/bin/python scripts/demo_experiment.py    # Task 8: multi-start table
.venv/bin/python scripts/demo_sweeps.py        # Task 9: sensitivity sweeps
```

## Tests

```bash
.venv/bin/pytest
```

## Architecture

Data flows **down** the layers below; nothing lower ever imports something
higher. `geometry.py` doesn't know the solvers exist, and `config.py` depends on
nothing. This one-way dependency flow is what keeps the code modular and the
comparison fair.

```
config.py          parameters — depends on nothing
   |
problem.py         builds region M + sampled points P from a Config
   |
geometry.py        distances, coverage, connectivity — the computational core
   |
   +-- pso.py      solver 1 (gradient-free): uses geometry
   +-- sqp.py      solver 2 (gradient-based): uses geometry
        |
experiment.py      runs both solvers many times, collects + validates metrics
   |
visualize.py       turns problems/results into PNGs
   |
main.py            CLI: wires experiment + visualize into one pipeline

scripts/demo_*.py  one standalone demo per task (same building blocks)
tests/test_*.py    pytest checks mirroring each module
```

### File-by-file

| File                | Responsibility                                              |
| ------------------- | ----------------------------------------------------------- |
| `wsn/config.py`     | Frozen `Config` dataclass: every parameter + seed. Immutable; use `with_overrides(...)` to get a modified copy (keeps sweeps isolated and runs reproducible). `validate()` catches bad setups. Depends on nothing. |
| `wsn/problem.py`    | `build_problem(config)` turns the "recipe" (Config) into "ingredients" (a `Problem`): the sampled grid `P`, base-station coords, and helpers `random_layout()` / `clip_to_region()`. Solvers receive a `Problem`; sensor positions `S` are passed separately as the decision variable. |
| `wsn/geometry.py`   | The math core, vectorized with NumPy (it's the inner loop). **Hard/true** functions: `coverage_ratio` (objective), `is_connected` (BFS over the comm graph). **Smooth/surrogate** functions for SQP: `soft_coverage_ratio` (sigmoid) and `algebraic_connectivity` (Fiedler λ₂). True and smooth live side-by-side on purpose — the surrogate-vs-truth relationship is the crux of the project. |
| `wsn/pso.py`        | Hand-written PSO (gradient-free). Scores layouts via `geometry`. Two constraint modes: `penalty` (lowers fitness of disconnected layouts) and `repair` (pulls stranded sensors back in range). `solve_pso()` reads `config.pso_constraint_mode` and dispatches. Returns a `PSOResult`. |
| `wsn/sqp.py`        | Wraps SciPy `SLSQP` (gradient-based, local). Minimizes `-soft_coverage` subject to region bounds and `λ₂ ≥ ε`. `run_sqp()` runs the **continuation** loop (increasing β, warm-started). Home of the `sqp_comm_margin` fix for "smooth-feasible but truly disconnected". Returns an `SQPResult`. |
| `wsn/experiment.py` | The orchestrator. Imports **both** solvers and runs them over methods × seeds. `_run_one` always **re-validates with the true checks**, never the surrogate. `run_experiment` aggregates into `MethodSummary`. `sweep_initial_conditions` / `sweep_parameter` drive the sensitivity analysis. |
| `wsn/visualize.py`  | Pure rendering (Matplotlib `Agg` backend, never blocks). Consumes `Problem` + result/summary objects and writes PNGs: coverage maps, convergence curves, seed-spread box plot, parameter sweep, comparison table. |
| `wsn/__init__.py`   | Package front door: re-exports common names so callers can `from wsn import build_problem`. |
| `main.py`           | End-to-end CLI. Parses flags into a `Config`, then: `run_experiment` → print table → plots → layout maps → sweeps. Imports only `experiment` + `visualize` (solvers are abstracted away behind `experiment`). |
| `scripts/demo_*.py` | One standalone demo per implementation task — exercises a single concept in isolation using the same `wsn` building blocks. |
| `tests/test_*.py`   | pytest files mirroring the modules; encode the correctness properties (coverage math, repair reconnects, soft → hard as β grows, pipeline runs end to end). |
| `conftest.py`       | Puts the project root on `sys.path` so `import wsn` works under pytest regardless of install state. |
| `pyproject.toml`    | Package metadata, dependencies, and pytest config. |

### What happens on `python main.py`

1. `main.py` builds a `Config`, calls `build_problem()` → a `Problem` (region + points).
2. It calls `experiment.run_experiment(config)`.
3. For each method × seed, `experiment` calls `pso.solve_pso()` or `sqp.run_sqp()`.
4. Those solvers repeatedly call `geometry` functions to score candidate layouts.
5. `experiment` re-checks each result with the **true** `geometry` functions and packages `MethodSummary` objects.
6. `main.py` hands those summaries to `visualize`, which writes PNGs to `outputs/`.

So `geometry` is the hub everyone leans on, `experiment` is the conductor, and
`main` is just the button that starts it.

> Note: `pso.py` and `sqp.py` are **siblings** — neither imports the other, they
> only share `geometry`. That's what lets `experiment` treat them as
> interchangeable methods behind one interface.

## Key findings (default instance)

| method      | best C | mean C | feasible % | notes                              |
| ----------- | ------ | ------ | ---------- | ---------------------------------- |
| PSO-penalty | ~0.73  | ~0.71  | 100%       | robust, always feasible            |
| PSO-repair  | ~0.74  | ~0.73  | 100%       | best mean, tightest spread         |
| SQP         | ~0.76  | ~0.73  | ~40%       | highest peak, often infeasible     |

**Interpretation.** The coverage objective is highly **multimodal** and the
feasible region (connected layouts) is **non-convex and often disconnected**.
SQP, a local gradient method, can reach the highest coverage from a lucky start
but is **sensitive to initialization** and frequently converges to **infeasible
local optima** (the smooth λ₂ surrogate can be satisfied while the true graph is
marginally disconnected — a safety margin on the effective comm radius mitigates
but does not eliminate this). PSO explores the space more broadly and reliably
returns **feasible** layouts, at the cost of a slightly lower peak. This is the
expected population-vs-gradient trade-off for nonlinear, constrained placement.
