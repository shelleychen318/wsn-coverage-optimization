"""WSN area-coverage maximization under network connectivity.

A small research codebase comparing Particle Swarm Optimization (PSO) and
Sequential Quadratic Programming (SQP) on the sensor-placement problem.
"""

from .config import Config, DEFAULT_CONFIG
from .problem import Problem, build_problem

__all__ = ["Config", "DEFAULT_CONFIG", "Problem", "build_problem"]
