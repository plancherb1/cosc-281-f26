"""Released public tests; staff-only checks are omitted."""
import math

import numpy as np

import pytest

import gradelib

def _grid(art):
    """String art to occupancy: '#' = obstacle, '.' = free."""
    return np.array([[ch == "#" for ch in row] for row in art])

def _astar():
    """The student's astar, imported the one sanctioned way."""
    return gradelib.load("astar").astar

def _cells(seq):
    """Normalize a cell sequence to plain int tuples for == pinning."""
    return [tuple(int(v) for v in cell) for cell in seq]

def _check_path_structure(result, occ, start, goal):
    """The structural path contract: start..goal inclusive, 4-connected
    unit steps, never through an obstacle, length == cost + 1."""
    path = _cells(result.path)
    assert path, "A reachable goal returned an empty path. Check TODO A: update g, parent, and the frontier on a cheaper route."
    assert path[0] == tuple(start) and path[-1] == tuple(goal), \
        f"Path must include start {start} and goal {goal}; got endpoints {path[0]} and {path[-1]}."
    assert len(path) == int(result.cost) + 1, \
        "Path cost counts moves, not cells: it must equal len(path) - 1. Check the unit edge cost in TODO A."
    n_rows, n_cols = occ.shape
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        assert abs(r0 - r1) + abs(c0 - c1) == 1, "path step not 4-connected"
    for r, c in path:
        assert 0 <= r < n_rows and 0 <= c < n_cols, "path leaves the grid"
        assert not occ[r, c], "path passes through an obstacle"

def _check_expansion_order(actual, expected):
    """Report the first divergent pop instead of printing two long lists."""
    actual = _cells(actual)
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert got == want, (
            f"A* expansion {i + 1}: expected cell {want}, got {got}. "
            "Positions count valid pops, starting at 1. Check TODO A: "
            "update only for a strictly cheaper route and push (g+h, h, cell). "
            "Keep the supplied frontier and stale-entry handling unchanged.")
    assert len(actual) == len(expected), (
        f"A* expansion sequence has {len(actual)} cells; expected {len(expected)}. "
        f"First missing or extra position: {min(len(actual), len(expected)) + 1}. "
        "Include start and goal, and do not count obsolete frontier entries.")

OPEN5 = ["....."] * 5

OPEN5_ORDER = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
               (1, 4), (2, 4), (3, 4), (4, 4)]

@pytest.mark.public
def test_open_grid_cost_and_path():
    occ = _grid(OPEN5)
    result = _astar()(occ, (0, 0), (4, 4))
    assert result.cost == 8, f"Open-grid path cost: expected 8 moves, got {result.cost}. Check TODO A's candidate cost and updates to g and parent."
    _check_path_structure(result, occ, (0, 0), (4, 4))

@pytest.mark.public
def test_open_grid_expansion_order():
    result = _astar()(_grid(OPEN5), (0, 0), (4, 4))
    _check_expansion_order(result.expanded, OPEN5_ORDER)

WALL5 = ["..#..",
         "..#..",
         "..#..",
         "..#..",
         "....."]

WALL5_ORDER = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 1), (2, 0), (3, 1),
               (3, 0), (4, 1), (4, 2), (4, 3), (3, 3), (2, 3), (1, 3),
               (0, 3), (0, 4)]

@pytest.mark.public
def test_walled_grid_cost_and_path():
    occ = _grid(WALL5)
    result = _astar()(occ, (0, 0), (0, 4))
    assert result.cost == 12, f"Walled-grid path cost: expected 12 moves, got {result.cost}. Check that TODO A updates an already discovered cell when a strictly cheaper route is found."
    _check_path_structure(result, occ, (0, 0), (0, 4))

@pytest.mark.public
def test_walled_grid_expansion_order():
    result = _astar()(_grid(WALL5), (0, 0), (0, 4))
    _check_expansion_order(result.expanded, WALL5_ORDER)

UNREACH5 = ["..#.."] * 5

UNREACH5_ORDER = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 1), (2, 0),
                  (3, 1), (3, 0), (4, 1), (4, 0)]

@pytest.mark.public
def test_unreachable_goal():
    result = _astar()(_grid(UNREACH5), (0, 0), (0, 4))
    assert result.path == []
    assert math.isinf(result.cost)
    _check_expansion_order(result.expanded, UNREACH5_ORDER)

@pytest.mark.public
def test_start_equals_goal():
    # Contract corner: the start expands first and the goal test happens
    # at pop, so start == goal means one expansion, cost 0, path [start].
    result = _astar()(_grid(OPEN5), (2, 2), (2, 2))
    assert result.cost == 0
    assert _cells(result.path) == [(2, 2)]
    assert _cells(result.expanded) == [(2, 2)]

@pytest.mark.public
def test_deterministic_across_runs():
    # No RNG anywhere in A* + DeterministicFrontier: two fresh imports
    # and runs must agree exactly, path included.
    first = _astar()(_grid(WALL5), (0, 0), (0, 4))
    second = _astar()(_grid(WALL5), (0, 0), (0, 4))
    assert _cells(first.path) == _cells(second.path)
    assert first.cost == second.cost
    assert _cells(first.expanded) == _cells(second.expanded)

