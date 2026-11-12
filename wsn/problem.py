"""Problem definition: the rectangular region M and sampled point set P.

The :class:`Problem` object bundles everything a solver needs that does not
change during optimization: the sampled points, the base station, the radii,
and the region bounds. Sensor positions S are the decision variables and are
passed into evaluation functions separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Problem:
    """A concrete WSN placement problem built from a :class:`Config`."""

    config: Config
    points: np.ndarray  # shape (|P|, 2): sampled points P over M
    base_station: np.ndarray  # shape (2,)

    # --- convenience passthroughs -------------------------------------------
    @property
    def n_sensors(self) -> int:
        return self.config.n_sensors

    @property
    def sensing_radius(self) -> float:
        return self.config.sensing_radius

    @property
    def comm_radius(self) -> float:
        return self.config.comm_radius

    @property
    def region(self):
        return self.config.region

    @property
    def bounds(self):
        """Per-variable (min, max) bounds for a flattened sensor vector.

        Order matches ``S.reshape(-1)`` for S of shape (n_sensors, 2):
        [x0, y0, x1, y1, ...].
        """
        x_min, y_min, x_max, y_max = self.config.region
        return [(x_min, x_max), (y_min, y_max)] * self.n_sensors

    def clip_to_region(self, sensors: np.ndarray) -> np.ndarray:
        """Clamp sensor coordinates so they stay inside M."""
        x_min, y_min, x_max, y_max = self.config.region
        clipped = np.array(sensors, dtype=float, copy=True)
        clipped[:, 0] = np.clip(clipped[:, 0], x_min, x_max)
        clipped[:, 1] = np.clip(clipped[:, 1], y_min, y_max)
        return clipped

    def random_layout(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a uniform random sensor layout inside M, shape (n_sensors, 2)."""
        x_min, y_min, x_max, y_max = self.config.region
        xs = rng.uniform(x_min, x_max, size=self.n_sensors)
        ys = rng.uniform(y_min, y_max, size=self.n_sensors)
        return np.column_stack([xs, ys])


def build_problem(config: Config) -> Problem:
    """Construct a :class:`Problem`: discretize M into the sampled grid P."""
    config.validate()
    x_min, y_min, x_max, y_max = config.region
    res = config.grid_resolution

    # Cell-centered grid so points stay strictly inside M.
    xs = np.linspace(x_min, x_max, res)
    ys = np.linspace(y_min, y_max, res)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    base = np.asarray(config.base_station, dtype=float)
    return Problem(config=config, points=points, base_station=base)
