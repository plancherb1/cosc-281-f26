"""Provided search utilities for PS1.

Implement A* in astar.py; no changes to this file are required. Grading
uses the supplied version of these utilities.

Grid convention: a cell is a (row, col) pair of integers,
row 0 at the top of the grid, col 0 at the left. An occupancy grid is an
HxW numpy bool array with True = obstacle. All moves are 4-connected
(up, down, left, right) and cost 1 each, so costs and heuristic values
are in units of moves.
"""
import heapq
from dataclasses import dataclass

Cell = tuple  # (row, col) ints; see the grid convention above


def manhattan(a, b):
    """Manhattan distance |dr| + |dc| between two cells, in moves.

    The default heuristic. It is admissible and consistent on a
    4-connected unit-cost grid.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class SearchResult:
    """Search path, cost, and expansion sequence returned by astar().

    path      list of cells including start and goal; [] if the goal
              is unreachable.
    cost      float, path length in moves (len(path) - 1 on unit grids);
              math.inf if unreachable.
    expanded  list of cells in the order they were expanded (popped from
              the frontier as non-stale entries), goal included.
    """
    path: list
    cost: float
    expanded: list


class DeterministicFrontier:
    """Min-heap frontier ordered by (f, h, row, col): the course tie-break.

    Ties in f break toward lower h; ties in (f, h) break row-major, so
    smaller row first, then smaller column. Distinct cells have distinct
    keys, so their removal order does not depend on insertion order.

    The frontier supports duplicate entries through lazy deletion. When
    a lower-cost route to a cell is found, insert a new entry with the
    improved f value. The previous entry remains in the heap; the supplied
    A* loop identifies and skips obsolete entries when they are removed.
    """

    def __init__(self):
        self._heap = []

    def push(self, f, h, cell):
        """Add an entry. f = g + h is the priority, h is the heuristic
        value used for tie-breaking, cell is (row, col)."""
        heapq.heappush(self._heap, (f, h, cell[0], cell[1]))

    def pop(self):
        """Remove and return the smallest entry as (f, h, cell).
        Raises IndexError if the frontier is empty."""
        f, h, row, col = heapq.heappop(self._heap)
        return f, h, (row, col)

    def __len__(self):
        return len(self._heap)
