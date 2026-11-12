"""Vectorized geometry: distances, coverage (hard + soft), and connectivity.

This module is the computational inner loop. Coverage is evaluated thousands
of times by both solvers, so everything is vectorized with NumPy.

Conventions
-----------
- ``sensors`` is an array of shape (N, 2).
- ``points`` is an array of shape (P, 2) (the sampled set from the Problem).
- Distances are Euclidean.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of ``a`` and every row of ``b``.

    Returns an array of shape (len(a), len(b)).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ---------------------------------------------------------------------------
# Hard (true) coverage
# ---------------------------------------------------------------------------
def coverage_mask(sensors: np.ndarray, points: np.ndarray, rs: float) -> np.ndarray:
    """Boolean array of shape (P,): True if a point is covered by >=1 sensor."""
    dist = pairwise_distances(sensors, points)  # (N, P)
    covered = dist <= rs  # (N, P)
    return np.any(covered, axis=0)  # (P,)


def coverage_ratio(sensors: np.ndarray, points: np.ndarray, rs: float) -> float:
    """True coverage ratio C(S): fraction of sampled points covered.

    Overlap is not double-counted (a point counts once regardless of how many
    sensors cover it), which is exactly the ``max_i f(s_i, p)`` semantics.
    """
    if len(points) == 0:
        return 0.0
    return float(np.mean(coverage_mask(sensors, points, rs)))


# ---------------------------------------------------------------------------
# Smoothing primitive
# ---------------------------------------------------------------------------
def sigmoid(x: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable logistic sigmoid of ``beta * x``.

    Used both for the soft coverage indicator and for smooth edge weights in
    the algebraic-connectivity surrogate.
    """
    z = beta * np.asarray(x, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# Communication graph / connectivity
# ---------------------------------------------------------------------------
def all_nodes(sensors: np.ndarray, base_station: np.ndarray) -> np.ndarray:
    """Stack sensors and the base station into one (N+1, 2) node array.

    The base station is the LAST node (index N).
    """
    base = np.asarray(base_station, dtype=float).reshape(1, 2)
    return np.vstack([np.asarray(sensors, dtype=float), base])


def comm_adjacency(sensors: np.ndarray, base_station: np.ndarray, rc: float) -> np.ndarray:
    """Boolean adjacency matrix of the communication graph (hard edges).

    Nodes are [sensors..., base_station]. An edge exists between two distinct
    nodes whose Euclidean distance is <= rc. Shape (N+1, N+1).
    """
    nodes = all_nodes(sensors, base_station)
    dist = pairwise_distances(nodes, nodes)
    adj = dist <= rc
    np.fill_diagonal(adj, False)
    return adj


def comm_edges(sensors: np.ndarray, base_station: np.ndarray, rc: float):
    """Return the list of (i, j) edges (i < j) of the communication graph."""
    adj = comm_adjacency(sensors, base_station, rc)
    iu, ju = np.triu_indices_from(adj, k=1)
    return [(int(i), int(j)) for i, j in zip(iu, ju) if adj[i, j]]


def is_connected(sensors: np.ndarray, base_station: np.ndarray, rc: float) -> bool:
    """True iff every node can reach the base station (direct or multi-hop).

    Uses a connected-components analysis over the sparse communication graph.
    This is the *true* (non-smooth) feasibility check used by PSO and for
    final validation.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = comm_adjacency(sensors, base_station, rc)
    n_components, _ = connected_components(csr_matrix(adj), directed=False)
    return n_components == 1


def algebraic_connectivity(
    sensors: np.ndarray, base_station: np.ndarray, rc: float, beta: float
) -> float:
    """Smooth algebraic connectivity (Fiedler value) lambda_2 of the graph.

    Edge weights are smooth: w_ij = sigmoid(beta * (rc - d_ij)), in (0, 1).
    We form the weighted Laplacian L = D - W and return its second-smallest
    eigenvalue. lambda_2 > 0 corresponds to a (weighted-)connected graph, so
    the SQP constraint lambda_2 >= eps pushes toward connected layouts with a
    differentiable signal.
    """
    nodes = all_nodes(sensors, base_station)
    dist = pairwise_distances(nodes, nodes)
    weights = sigmoid(rc - dist, beta)  # (N+1, N+1), in (0, 1)
    np.fill_diagonal(weights, 0.0)
    degree = np.sum(weights, axis=1)
    laplacian = np.diag(degree) - weights

    # Symmetric Laplacian -> real eigenvalues; take the second smallest.
    eigvals = np.linalg.eigvalsh(laplacian)
    return float(eigvals[1])


# ---------------------------------------------------------------------------
# Soft (differentiable) coverage  (Task 6)
# ---------------------------------------------------------------------------
def soft_coverage_ratio(
    sensors: np.ndarray, points: np.ndarray, rs: float, beta: float
) -> float:
    """Differentiable approximation of the coverage ratio C(S).

    Each (sensor, point) pair contributes a soft indicator
        g_ip = sigmoid(beta * (rs - d_ip))  in (0, 1).
    The per-point coverage probability uses a smooth soft-OR (noisy-OR):
        c_p = 1 - prod_i (1 - g_ip),
    i.e. a point is "covered" if at least one sensor covers it, without
    double-counting overlap. The ratio is the mean of c_p over all points.

    As beta -> infinity, sigmoid -> step and soft_coverage_ratio ->
    coverage_ratio (the true Boolean metric). This is the objective SQP
    maximizes, sharpened across the continuation schedule.
    """
    if len(points) == 0:
        return 0.0
    dist = pairwise_distances(sensors, points)  # (N, P)
    g = sigmoid(rs - dist, beta)  # (N, P) in (0, 1)
    # Product over sensors of (1 - g); use log-space for stability.
    log_not = np.sum(np.log1p(-g + 1e-15), axis=0)  # (P,)
    not_covered = np.exp(log_not)  # prod_i (1 - g_ip)
    c_p = 1.0 - not_covered
    return float(np.mean(c_p))
