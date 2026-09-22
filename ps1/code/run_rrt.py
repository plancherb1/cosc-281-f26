"""Provided runner for PS1 Task 3: RRT on the course maps.

Run from the repository root with cs281venv activated:

    python ps1/code/run_rrt.py --plot-dir ps1/figures

This runner calls rrt() from rrt.py on the three course maps with the supplied
constants and seed, prints samples drawn / nodes added per map, and
draws each tree with its edges as the actual Dubins arcs. The narrow
map's Task 3 figure is ps1_rrt.png; the other maps go
to ps1_rrt_<map>.png.
Submit only the narrow plot and the short sample-to-node-ratio response.
The other plots and the full printed table are for inspection.

On the maze map, the car plans to the specified checkpoint below the goal
chamber, because the chamber's only opening is a 0.4 m corridor the
car cannot fit through with this task's clearance. The runner supplies
the checkpoint as world.goal; use that goal without modification.

--save-artifacts resamples each found path to (M, 2) world waypoints
(<= 0.25 m apart) in out/rrt_path_<map>.npy. This optional export is not
required for PS1.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
try:
    import coursesim  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "sim"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from coursesim.viz import MapView

import constants
from dubins import DubinsWorld, edge_arc, path_waypoints
from rrt import rrt

MAPS = ("open", "maze", "narrow")


def plan_map(name):
    """One seeded RRT run on a course map. Returns (world, result)."""
    world = DubinsWorld(name)
    rng = np.random.default_rng(constants.RRT_SEED)
    result = rrt(world, world.start, world.goal, rng, constants.RRT_STEP,
                 constants.RRT_GOAL_BIAS, constants.RRT_GOAL_TOL,
                 constants.RRT_MAX_ITERS)
    return world, result


def save_tree_plot(world, result, path, title):
    """Tree with arc edges, path highlighted, start/goal marked."""
    view = MapView(world.map)
    edges = [edge_arc(result.tree[parent][0], state)
             for state, parent in result.tree[1:]]
    view.add_tree(edges)
    if result.path is not None:
        view.add_path(path_waypoints(result.path, spacing=0.1),
                      color="tab:red", linewidth=2.2)
    fig, ax = view.render()
    ax.plot(world.goal[0], world.goal[1], marker="x", markersize=10,
            color="tab:red", zorder=6)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save-artifacts", action="store_true",
                    help="write out/rrt_path_<map>.npy waypoint files "
                         "(optional; not submitted)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "out"))
    ap.add_argument("--plot-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"{'map':<9}{'found':>6}{'samples':>9}{'nodes':>7}"
          f"{'nodes/sample':>14}")
    for name in MAPS:
        world, result = plan_map(name)
        found = result.path is not None
        ratio = result.nodes_added / max(result.samples_drawn, 1)
        print(f"{name:<9}{str(found):>6}{result.samples_drawn:>9}"
              f"{result.nodes_added:>7}{ratio:>14.3f}")
        if name == "maze":
            print("  (maze plans to the checkpoint below the goal "
                  "chamber; see constants.py)")
        fname = "ps1_rrt.png" if name == "narrow" else f"ps1_rrt_{name}.png"
        plot_path = os.path.join(args.plot_dir, fname)
        save_tree_plot(world, result, plot_path,
                       f"{name}: Dubins RRT, "
                       f"{result.samples_drawn} iterations")
        print(f"  wrote {plot_path}")
        if args.save_artifacts and found:
            os.makedirs(args.out_dir, exist_ok=True)
            wp = path_waypoints(result.path)
            np.save(os.path.join(args.out_dir, f"rrt_path_{name}.npy"), wp)
            print(f"  wrote {args.out_dir}/rrt_path_{name}.npy "
                  f"({len(wp)} waypoints)")


if __name__ == "__main__":
    main()
