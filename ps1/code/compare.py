"""Provided comparison runner for PS1.

Run python ps1/code/compare.py --plot-dir ps1/figures from the course root.
Complete Task 1 first: this runner calls your A* as well as VI and PI.
It compares A*, value iteration, and policy iteration on the three course
maps, and saves value tables and ps1_values.png. Counts refer to different operations; timings
include planning and policy extraction, but not plotting or loading maps.
Submit only the three maze rows and the short timing interpretation.
The value plot and saved arrays are for inspection, not submission.
"""
import argparse
import json
import os
import sys
import time

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
from grid_mdp import ACTIONS, GridMDP, greedy_rollout
from policy_iteration import policy_iteration
from astar import astar
from coursesim import Map
from grid_mdp import map_path
from vi import extract_policy, value_iteration

MAPS = ("open", "maze", "narrow")

# World-frame (dx, dy) per action index, for the policy arrows.
_ARROW = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}


def run_map(name):
    world = GridMDP.from_map(name)
    start = time.perf_counter()
    V, sweeps = value_iteration(world, constants.GAMMA, constants.VI_TOL,
                               constants.VI_MAX_SWEEPS)
    policy = extract_policy(world, V, constants.GAMMA)
    vi_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    V_pi, pi, rounds = policy_iteration(world, constants.GAMMA)
    pi_ms = 1000 * (time.perf_counter() - start)
    np.testing.assert_allclose(V_pi, V, atol=1e-7, equal_nan=True,
        err_msg="VI and PI disagree. Run the Task 2 public tests; check TODO V's terminal backup and TODO P/I's greedy action selection before reporting timings.")
    m = Map.load(map_path(name))
    start_cell = tuple(int(v) for v in m.world_to_grid(m.start[0], m.start[1]))
    goal_cell = tuple(int(v) for v in m.world_to_grid(m.goal[0], m.goal[1]))
    occupancy = m.occupancy_grid(inflated=True)
    start = time.perf_counter()
    search = astar(occupancy, start_cell, goal_cell)
    astar_ms = 1000 * (time.perf_counter() - start)
    rows = [
        ('A* (0.05m)', f'{search.cost * m.resolution:.1f}',
         f'{len(search.expanded)} expansions', astar_ms),
        ('VI (1.0m)', _fmt_steps(greedy_rollout(world, policy, 1000), world),
         f'{sweeps} sweeps', vi_ms),
        ('PI (1.0m)', _fmt_steps(greedy_rollout(world, pi, 1000), world),
         f'{rounds} rounds', pi_ms),
    ]
    return world, V, policy, rows


def save_values_plot(world, V, policy, path):
    """Plot maze values and policy arrows for inspection, not submission."""
    view = MapView(_course_map(world))
    view.add_heatmap(V, cmap="viridis", alpha=0.65)
    fig, ax = view.render()
    for row, col in world.free_cells():
        if (row, col) == world.goal:
            continue
        a = int(policy[row, col])
        if a < 0:
            continue
        x, y = world.cell_center_world(row, col)
        dx, dy = _ARROW[a]
        scale = 0.32 * world.cell_size
        ax.annotate("", xy=(x + dx * scale, y + dy * scale), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.1))
    mappable = ax.images[-1]
    fig.colorbar(mappable, ax=ax, label="V(s)")
    ax.set_title(f"maze: converged V, gamma = {constants.GAMMA}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _course_map(world):
    from coursesim import Map
    from grid_mdp import map_path
    # world remembers only origin/cell size; reload the maze map for the
    # background. This is used only for the optional maze value plot.
    return Map.load(map_path("maze"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "out"),
                    help="where the values_*.npy artifacts go")
    ap.add_argument("--plot-dir", default=".",
                    help="where ps1_values.png goes (default: cwd)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    os.makedirs(args.plot_dir, exist_ok=True)
    print(f"{'map':<9}{'method':<13}{'path (m)':>9}{'work count':>22}{'time (ms)':>13}")
    for name in MAPS:
        world, V, policy, rows = run_map(name)
        for i, (method, dist, work, ms) in enumerate(rows):
            label = name if i == 0 else ""
            print(f"{label:<9}{method:<13}{dist:>9}{work:>22}{ms:>13.2f}")
        np.save(os.path.join(args.out_dir, f"values_{name}.npy"), V)
        if name == "maze":
            plot_path = os.path.join(args.plot_dir, "ps1_values.png")
            save_values_plot(world, V, policy, plot_path)
            print(f"  wrote {plot_path}")
    print(f"wrote {args.out_dir}/values_{{open,maze,narrow}}.npy "
          "(saved planning results)")


def _fmt_steps(steps, world):
    if steps is None:
        return "stuck"
    return f"{steps * world.cell_size:.1f}"


if __name__ == "__main__":
    main()
