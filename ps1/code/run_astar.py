"""Provided runner for PS1 Task 1: A* on the course maps.

Run from the repository root with cs281venv activated:

    python ps1/code/run_astar.py --out-dir ps1/figures

For each named map (sim/maps/<name>.yaml) this loads the world, inflates
obstacles by the simulated car's footprint radius, converts the start
and goal poses to grid cells, and passes the resulting NumPy Boolean
array to astar(). It prints the results table (path
cost in moves, nodes expanded) and writes ps1_astar_<name>.png per map:
expanded cells shaded in pop order, your path drawn over the obstacles.
Inspect all three images. Include only the maze image in your PDF and
report the open/maze expansion counts in the short response; no table
or other A* image is required.
"""
import argparse
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
try:
    import coursesim  # noqa: F401  (installed or already on the path)
except ImportError:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "sim"))

import numpy as np

from coursesim import Map
from coursesim.viz import MapView

from astar import astar  # your implementation


def solve_map(yaml_path):
    """Run the student astar() on one map.

    Returns (map, occ, start_cell, goal_cell, SearchResult). occ is the
    footprint-inflated occupancy grid (HxW bool, True = obstacle); cells
    are (row, col). The autograder performs this exact conversion.
    """
    m = Map.load(yaml_path)
    occ = m.occupancy_grid(inflated=True)
    start = tuple(int(v) for v in m.world_to_grid(m.start[0], m.start[1]))
    goal = tuple(int(v) for v in m.world_to_grid(m.goal[0], m.goal[1]))
    return m, occ, start, goal, astar(occ, start, goal)


def save_overlay(m, result, out_path):
    """Overlay plot: expanded cells shaded by pop order, path on top."""
    view = MapView(m)
    shade = np.full(m.shape, np.nan)
    for order, (row, col) in enumerate(result.expanded):
        shade[row, col] = order
    view.add_heatmap(shade, cmap="plasma", alpha=0.5)
    if result.path:
        rows = np.array([c[0] for c in result.path])
        cols = np.array([c[1] for c in result.path])
        xs, ys = m.grid_to_world(rows, cols)
        view.add_path(np.column_stack([xs, ys]))
    view.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--maps", nargs="+", default=["open", "maze", "narrow"],
                    help="map names under sim/maps/ (default: the three "
                         "course maps)")
    ap.add_argument("--out-dir", default=".",
                    help="where the overlay images go (default: cwd)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'map':<12}{'cost':>8}{'expanded':>10}")
    for name in args.maps:
        yaml_path = os.path.join(_REPO_ROOT, "sim", "maps", name + ".yaml")
        if not os.path.exists(yaml_path):
            sys.exit(f"no such course map: {yaml_path}")
        m, occ, start, goal, result = solve_map(yaml_path)
        cost = "inf" if math.isinf(result.cost) else f"{int(result.cost)}"
        print(f"{name:<12}{cost:>8}{len(result.expanded):>10}")
        out_path = os.path.join(args.out_dir, f"ps1_astar_{name}.png")
        save_overlay(m, result, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
