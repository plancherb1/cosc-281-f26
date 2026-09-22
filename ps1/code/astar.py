"""PS1 Task 1: A* on an occupancy grid.

Complete TODO A in astar() using the following search conventions:

  * Moves are 4-connected (up, down, left, right), each costing 1 move.
  * Test for the goal when you remove it from the frontier, not when you
    generate it as a neighbor.
  * An expansion is a pop of a non-stale entry. Record every expansion,
    in order, in the result's `expanded` list. The start is expanded
    first and the goal is included.
  * `path` runs from start to goal inclusive; `cost` is its length in
    moves. If the goal is unreachable, return an empty path with cost
    math.inf and the expansion sequence recorded before exhaustion.
  * Use the provided DeterministicFrontier. It enforces the course
    tie-breaking rule (f, then h, then row-major). Duplicate pushes for
    the same cell are allowed; skip stale entries when they pop.

run_astar.py loads maps, calls astar(), and plots the returned result.
compare.py also calls astar() after running VI and PI. You do not need
to edit either runner or implement plotting, map loading, or a heap.

Complete TODO A below: update a neighbor when a lower-cost route is found,
then add it to the frontier with priority f = g + h. Here g is the cost
from the start along that route, and h estimates the remaining cost to
the goal. Initialization, stale-entry checks, neighbor filtering, expansion
recording, and path recovery are provided. Read those sections to see
how your updates to g, parent, and frontier are used.
"""
import math

import numpy as np

from search_utils import DeterministicFrontier, SearchResult, manhattan


def astar(occ, start, goal, heuristic=manhattan):
    """4-connected A* on a unit-cost grid.

    Args:
        occ: An (H, W) NumPy Boolean array; True denotes an obstacle.
        start: The initial cell as a (row, col) integer tuple.
        goal: The goal cell as a (row, col) integer tuple.
        heuristic: A callable(cell, goal) returning an estimated cost in
            moves. Use the supplied function rather than calling the
            default Manhattan heuristic directly.

    Returns:
        SearchResult(path, cost, expanded), as defined in search_utils.py.
    """
    occ = np.asarray(occ, dtype=bool)
    n_rows, n_cols = occ.shape

    # g records the best discovered cost to each cell; parent
    # records its predecessor. Missing cells have no known route yet.
    g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = []
    frontier = DeterministicFrontier()
    h0 = heuristic(start, goal)
    frontier.push(h0, h0, start)

    while len(frontier):
        f, h, cell = frontier.pop()
        # An obsolete heap entry can remain after a lower-cost insertion.
        # Neither an obsolete entry nor an already expanded cell counts.
        if cell in closed or f > g[cell] + heuristic(cell, goal):
            continue
        closed.add(cell)
        expanded.append(cell)

        # Check for the goal after recording a valid expansion.
        if cell == goal:
            path = []
            node = cell
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return SearchResult(path, float(g[cell]), expanded)

        row, col = cell
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (row + drow, col + dcol)
            if not (0 <= nb[0] < n_rows and 0 <= nb[1] < n_cols):
                continue
            if occ[nb]:
                continue

            # TODO A: update the route to nb and its A* priority.
            # The cost of reaching nb through cell is g[cell] + 1.
            # Compare this cost with g.get(nb, math.inf). If it is lower,
            # store it in g[nb] and set parent[nb] to cell. Then compute
            # h = heuristic(nb, goal) and add nb to the frontier using
            # frontier.push(g[nb] + h, h, nb): priority = cost so far +
            # estimated remaining cost. The second argument breaks ties.
            # Otherwise, leave g, parent, and frontier unchanged.
            raise NotImplementedError("TODO A: update neighbor cost and A* priority")

    # Exhausting the frontier without reaching the goal indicates failure.
    return SearchResult([], math.inf, expanded)
