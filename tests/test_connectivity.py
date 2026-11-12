"""Tests for Task 3: true connectivity check and smooth Fiedler surrogate."""

from __future__ import annotations

import numpy as np

from wsn.geometry import (
    algebraic_connectivity,
    comm_adjacency,
    comm_edges,
    is_connected,
)


BASE = np.array([5.0, 5.0])
RC = 3.0


def test_connected_layout_true_and_lambda2_positive():
    # Sensors in a chain close to the base station -> connected.
    sensors = np.array(
        [
            [5.0, 6.0],
            [5.0, 7.0],
            [6.0, 5.0],
            [7.0, 5.0],
        ]
    )
    assert is_connected(sensors, BASE, RC) is True
    lam2 = algebraic_connectivity(sensors, BASE, RC, beta=10.0)
    assert lam2 > 1e-3


def test_isolated_sensor_disconnected_and_lambda2_near_zero():
    # One sensor is far away in a corner, out of comm range of everything.
    sensors = np.array(
        [
            [5.0, 6.0],
            [6.0, 5.0],
            [0.0, 0.0],  # isolated
        ]
    )
    assert is_connected(sensors, BASE, RC) is False
    lam2 = algebraic_connectivity(sensors, BASE, RC, beta=10.0)
    assert lam2 < 1e-2


def test_lambda2_increases_as_nodes_move_closer():
    base = np.array([0.0, 0.0])
    rc = 3.0
    # Two sensors; move the second from just-in-range toward the base/first.
    far = np.array([[1.0, 0.0], [2.9, 0.0]])
    near = np.array([[1.0, 0.0], [1.5, 0.0]])
    lam_far = algebraic_connectivity(far, base, rc, beta=5.0)
    lam_near = algebraic_connectivity(near, base, rc, beta=5.0)
    assert lam_near > lam_far


def test_adjacency_and_edges_consistent():
    sensors = np.array([[5.0, 6.0], [5.0, 7.5]])
    adj = comm_adjacency(sensors, BASE, RC)
    # Symmetric, no self-loops.
    assert np.array_equal(adj, adj.T)
    assert not np.any(np.diag(adj))
    edges = comm_edges(sensors, BASE, RC)
    # Edge count equals number of True entries in upper triangle.
    assert len(edges) == int(np.sum(np.triu(adj, k=1)))


def test_base_station_is_last_node():
    sensors = np.array([[5.0, 6.0], [5.0, 7.0]])
    adj = comm_adjacency(sensors, BASE, RC)
    # N sensors + 1 base station.
    assert adj.shape == (3, 3)
