"""Central configuration for the WSN coverage-maximization project.

All tunable parameters live here so experiments and sensitivity sweeps can
override a single dataclass instead of threading arguments through every
module. Values are deliberately modest so the full pipeline runs quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Immutable configuration bundle for a single problem + solver setup.

    Use :meth:`with_overrides` to produce a modified copy (keeps the base
    config intact, which is convenient for parameter sweeps).
    """

    # ----- Problem definition ------------------------------------------------
    n_sensors: int = 8
    """Number of deployable sensor nodes N (excludes the base station)."""

    region: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)
    """Rectangular region M as (x_min, y_min, x_max, y_max)."""

    base_station: Tuple[float, float] = (5.0, 5.0)
    """Fixed base station location (x_b, y_b)."""

    sensing_radius: float = 1.8
    """Sensing radius r_s used for the coverage indicator."""

    comm_radius: float = 3.0
    """Communication radius r_c used for the connectivity graph."""

    grid_resolution: int = 40
    """Sampled points P are a grid_resolution x grid_resolution lattice over M."""

    # ----- Smoothing / continuation (SQP) ------------------------------------
    beta_schedule: Tuple[float, ...] = (1.0, 3.0, 8.0, 20.0, 50.0)
    """Increasing sigmoid sharpness values for the continuation (homotopy) loop."""

    lambda2_epsilon: float = 5e-2
    """Lower bound epsilon on the smooth algebraic connectivity constraint."""

    # ----- PSO parameters ----------------------------------------------------
    pso_swarm_size: int = 40
    pso_iterations: int = 120
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5
    pso_velocity_clamp_frac: float = 0.3
    """Max velocity per axis as a fraction of the region extent."""

    pso_penalty_weight: float = 2.0
    """Penalty coefficient gamma for disconnected layouts (penalty mode)."""

    pso_constraint_mode: str = "penalty"
    """Either 'penalty' or 'repair'."""

    # ----- SQP parameters ----------------------------------------------------
    sqp_max_iter: int = 200
    """Max iterations per inner SLSQP solve."""

    sqp_comm_margin: float = 0.4
    """Distance safety margin: the smooth connectivity constraint uses an
    effective comm radius (rc - margin). Because SLSQP tends to settle sensors
    on the active constraint boundary, this margin keeps the TRUE edges
    (distance <= rc) comfortably inside range, so a satisfied smooth
    constraint implies true connectivity."""

    # ----- Experiment / reproducibility --------------------------------------
    seed: int = 0
    """Master seed for a single run."""

    n_starts: int = 8
    """Number of random restarts used by the multi-start experiment runner."""

    output_dir: str = "outputs"
    """Directory where figures and tables are written."""

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy of this config with the given fields replaced."""
        return replace(self, **kwargs)

    # ----- Derived helpers ---------------------------------------------------
    @property
    def region_width(self) -> float:
        return self.region[2] - self.region[0]

    @property
    def region_height(self) -> float:
        return self.region[3] - self.region[1]

    @property
    def n_points(self) -> int:
        """Total number of sampled points |P|."""
        return self.grid_resolution * self.grid_resolution

    def validate(self) -> None:
        """Basic sanity checks; raises ValueError on an invalid config."""
        x_min, y_min, x_max, y_max = self.region
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("region must have positive width and height")
        if self.n_sensors < 1:
            raise ValueError("n_sensors must be >= 1")
        if self.sensing_radius <= 0 or self.comm_radius <= 0:
            raise ValueError("radii must be positive")
        if self.grid_resolution < 2:
            raise ValueError("grid_resolution must be >= 2")
        bx, by = self.base_station
        if not (x_min <= bx <= x_max and y_min <= by <= y_max):
            raise ValueError("base station must lie inside the region")
        if self.pso_constraint_mode not in ("penalty", "repair"):
            raise ValueError("pso_constraint_mode must be 'penalty' or 'repair'")


DEFAULT_CONFIG = Config()
