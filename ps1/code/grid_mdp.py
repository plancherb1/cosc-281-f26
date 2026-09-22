"""Provided grid Markov decision process for PS1.

Implement the planning algorithms in vi.py and policy_iteration.py.
No changes to this model are required; grading uses the supplied version.

The MDP is defined as follows:
  states   free cells (row, col) of a coarse occupancy grid laid over a
           course map. Row 0 is the top of the map (maximum world y),
           col 0 the left, matching the PS1 grid convention.
  actions  0=N, 1=S, 2=E, 3=W (greedy ties break
           toward the lower action index). N moves one row up (toward
           larger world y), E one column right (larger world x).
  moves    deterministic. Attempting to enter a wall or cross the grid edge is a
           self-transition: the state does not change.
  reward   0 on every step, +GOAL_REWARD (=10) on entering the goal
           cell. The goal is absorbing and terminal: the episode ends,
           and V(goal) = 0 by convention.
  discount GAMMA = 0.9.

The coarse grid: each MDP cell covers CELL_SIZE_M x CELL_SIZE_M meters
(1.0 m = 20 map pixels). A cell is free when the car's footprint
(CarParams().radius) fits at the cell center; this is the same collision
footprint convention used by the simulator. A* uses a finer occupancy
grid, while RRT plans continuous car motions and checks intermediate
states along each extension. See run_rrt.py for its planning clearances
and the maze checkpoint used in Task 3.
"""
import os

import numpy as np

from coursesim import CarParams, Map

import constants

# Action offsets in (drow, dcol), indexed 0..3 = N, S, E, W.
ACTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))
ACTION_NAMES = ("N", "S", "E", "W")

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def map_path(name):
    """Absolute path of a course map yaml under sim/maps/."""
    return os.path.join(_REPO_ROOT, "sim", "maps", name + ".yaml")


class GridMDP:
    """The PS1 grid MDP over an occupancy grid.

    Attributes (read-only by convention):
      occupancy  (H, W) bool array, True = wall.
      shape      (H, W).
      start      (row, col) start cell.
      goal       (row, col) goal cell, absorbing and terminal.
      actions    the ACTIONS tuple in N, S, E, W order.
      cell_size  meters per cell, used for coordinate conversion.
      origin     world (x, y) of the grid's lower-left corner, meters.
    """

    def __init__(self, occupancy, start, goal, cell_size=1.0,
                 origin=(0.0, 0.0)):
        self.occupancy = np.asarray(occupancy, dtype=bool)
        self.shape = self.occupancy.shape
        self.start = tuple(int(v) for v in start)
        self.goal = tuple(int(v) for v in goal)
        self.actions = ACTIONS
        self.cell_size = float(cell_size)
        self.origin = (float(origin[0]), float(origin[1]))
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if self.occupancy[cell]:
                raise ValueError(f"{name} cell {cell} is a wall")

    @classmethod
    def from_map(cls, name_or_path, cell_size=constants.CELL_SIZE_M):
        """Build the MDP for a course map ('open', 'maze', 'narrow').

        A coarse cell is free when Map.is_free holds at its center with
        the car's footprint radius. Start and goal are the cells
        containing the map's start and goal poses.
        """
        path = name_or_path if os.path.exists(str(name_or_path)) \
            else map_path(name_or_path)
        m = Map.load(path)
        radius = CarParams().radius
        h_px, w_px = m.occupancy.shape
        n_rows = int(h_px * m.resolution // cell_size)
        n_cols = int(w_px * m.resolution // cell_size)
        occ = np.ones((n_rows, n_cols), dtype=bool)
        for row in range(n_rows):
            for col in range(n_cols):
                x = m.origin[0] + (col + 0.5) * cell_size
                y = m.origin[1] + (n_rows - row - 0.5) * cell_size
                occ[row, col] = not m.is_free(x, y, radius=radius)

        def cell_of(pose):
            col = int((pose[0] - m.origin[0]) // cell_size)
            row = n_rows - 1 - int((pose[1] - m.origin[1]) // cell_size)
            return (row, col)

        return cls(occ, cell_of(m.start), cell_of(m.goal), cell_size,
                   (m.origin[0], m.origin[1]))

    # ------------------------------------------------------------ dynamics

    def step(self, s, a):
        """One MDP transition. s: (row, col) free cell. a: action index
        0..3 (N, S, E, W). Returns (s2, reward, done): s2 the next cell
        (== s on a bump), reward 0.0 or +10.0 on entering the goal, done
        True exactly when s2 is the goal. Deterministic, no RNG."""
        drow, dcol = ACTIONS[a]
        s2 = (s[0] + drow, s[1] + dcol)
        if not (0 <= s2[0] < self.shape[0] and 0 <= s2[1] < self.shape[1]) \
                or self.occupancy[s2]:
            s2 = s
        done = s2 == self.goal
        return s2, (constants.GOAL_REWARD if done else 0.0), done

    # ------------------------------------------------------------- helpers

    def free_cells(self):
        """List of all free (row, col) cells, row-major order."""
        return [tuple(c) for c in np.argwhere(~self.occupancy)]

    def cell_center_world(self, row, col):
        """Return the world coordinates (x, y) of a cell center in meters."""
        x = self.origin[0] + (col + 0.5) * self.cell_size
        y = self.origin[1] + (self.shape[0] - row - 0.5) * self.cell_size
        return x, y


def greedy_rollout(world, table, max_steps=200):
    """Roll the greedy policy out from world.start.

    table: an (H, W) integer array of action indices, from value iteration
    with policy extraction or from policy iteration.

    Returns the number of steps taken to enter the goal, or None if the
    rollout bumps in place or exceeds max_steps without reaching it.
    """
    table = np.asarray(table)
    s = world.start
    for k in range(1, max_steps + 1):
        a = int(table[s])
        s2, _, done = world.step(s, a)
        if s2 == s:
            return None            # bumping in place: policy is stuck
        s = s2
        if done:
            return k
    return None
