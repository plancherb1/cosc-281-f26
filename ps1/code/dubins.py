"""Provided Dubins-car world and motion primitives for PS1 Task 3.

Implement the RRT algorithm in rrt.py. No changes to this file are
required; grading uses the supplied dynamics and steering functions.

A state is a numpy (3,) array [x, y, theta] (meters, meters, radians),
following the simulator's pose convention. Motion is computed using
coursesim.dynamics.step. Each tree edge follows a simulated car motion:
maximum left steering, straight, or maximum right steering. The turning
radius at maximum steering is wheelbase / tan(max_steer) = 0.74 m.
Each motion primitive has arc length `step` meters.
"""
import math
import os
from dataclasses import dataclass

import numpy as np

from coursesim import CarParams, Map, dynamics

import constants
from grid_mdp import map_path

# Substep length (m) for collision checks along one primitive arc.
_COLLISION_SUBSTEP = 0.1


@dataclass
class RRTResult:
    """Path, tree, and sampling counts returned by rrt().

    path           list of (3,) states from the start to the node inside
                   the goal tolerance, including both endpoints; None
                   when the sampling budget is exhausted without success.
    tree           list of (state, parent_index) pairs in the order nodes
                   were added; the root is entry 0 with parent -1.
    samples_drawn  total samples drawn (goal-biased ones included).
    nodes_added    extensions that passed the collision check.
    """
    path: list
    tree: list
    samples_drawn: int
    nodes_added: int


class DubinsWorld:
    """Continuous course map + collision and sampling queries.

    Attributes:
      map     the loaded coursesim Map.
      car     CarParams (the course car; fixes the turning radius).
      radius  collision radius, m: car footprint plus the map's specified
              planning margin (constants.RRT_MARGIN).
      start   (3,) start pose from the map.
      goal    (3,) goal pose: the map goal, unless constants.RRT_GOALS
              specifies a checkpoint (see run_rrt.py for the maze case).
    """

    def __init__(self, name_or_path, margin=None):
        if isinstance(name_or_path, Map):
            # In-memory map (synthetic test worlds). Uses the default
            # margin unless one is passed explicitly.
            self.map = name_or_path
            name = None
        else:
            path = name_or_path if os.path.exists(str(name_or_path)) \
                else map_path(name_or_path)
            name = os.path.splitext(os.path.basename(str(path)))[0]
            self.map = Map.load(path)
        self.car = CarParams()
        if margin is None:
            margin = constants.RRT_MARGIN.get(name, 0.05)
        self.radius = self.car.radius + float(margin)
        self.start = self.map.start.copy()
        goal_override = constants.RRT_GOALS.get(name)
        self.goal = np.asarray(goal_override, dtype=float) \
            if goal_override is not None else self.map.goal.copy()

    def collision_free(self, state):
        """True when the collision disc (self.radius) centered at the
        state's (x, y) clears all obstacles. Theta does not matter for
        a disc footprint."""
        return bool(self.map.is_free(float(state[0]), float(state[1]),
                                     radius=self.radius))

    def sample_free(self, rng):
        """One uniform collision-free sample of the state space.

        Draw x, then y uniformly over the map extent, then theta over
        [-pi, pi); rejected samples repeat all three draws.
        """
        x0, x1, y0, y1 = self.map.extent
        while True:
            x = rng.uniform(x0, x1)
            y = rng.uniform(y0, y1)
            theta = rng.uniform(-math.pi, math.pi)
            if self.map.is_free(x, y, radius=self.radius):
                return np.array([x, y, theta])


def _primitive_end(state, steer, step, world=None):
    """Integrate one primitive arc of length `step` from `state`.

    Returns the end state, or None when `world` is given and any
    _COLLISION_SUBSTEP-spaced point along the arc collides."""
    n_sub = max(2, int(math.ceil(step / _COLLISION_SUBSTEP)))
    s = np.asarray(state, dtype=float)
    for _ in range(n_sub):
        s = dynamics.step(s, (1.0, steer), CarParams(), step / n_sub)
        if world is not None and not world.collision_free(s):
            return None
    return s


def dubins_steer(world, state, target, step):
    """Extend one primitive arc from `state` toward `target`.

    Evaluate the three motion primitives in the order maximum left
    (+max_steer), straight, maximum right (-max_steer), each of arc length
    `step` meters, discarding any whose arc collides (checked every
    0.1 m along the arc with the world's collision radius). Returns the
    surviving end state whose (x, y) is closest to the target's (a
    strict comparison, so the earlier primitive wins ties), or None
    when all three collide.

    The target's heading is not used in primitive selection. Distance
    is measured in (x, y), as in the RRT nearest-node calculation.
    """
    car = CarParams()
    best, best_dist = None, math.inf
    for steer in (car.max_steer, 0.0, -car.max_steer):
        end = _primitive_end(state, steer, step, world)
        if end is None:
            continue
        dist = math.hypot(end[0] - target[0], end[1] - target[1])
        if dist < best_dist:
            best, best_dist = end, dist
    return best


# ------------------------------------------------- plotting / artifacts

def match_primitive(parent, child, step=None):
    """Recover which primitive produced the tree edge parent -> child.

    Returns the steer command (+max_steer, 0.0, or -max_steer) whose
    arc endpoint reproduces `child` (closest match). Used by the plot
    and resampling helpers so the tree renders as the arcs the car
    would actually drive, not as chords.
    """
    if step is None:
        step = constants.RRT_STEP
    car = CarParams()
    best, best_dist = 0.0, math.inf
    for steer in (car.max_steer, 0.0, -car.max_steer):
        end = _primitive_end(parent, steer, step)
        dist = math.hypot(end[0] - child[0], end[1] - child[1])
        if dist < best_dist:
            best, best_dist = steer, dist
    return best


def edge_arc(parent, child, step=None, n_points=8):
    """(n_points+1, 2) polyline tracing the arc parent -> child."""
    if step is None:
        step = constants.RRT_STEP
    steer = match_primitive(parent, child, step)
    pts = [np.asarray(parent, dtype=float)[:2]]
    s = np.asarray(parent, dtype=float)
    for _ in range(n_points):
        s = dynamics.step(s, (1.0, steer), CarParams(), step / n_points)
        pts.append(s[:2])
    return np.array(pts)


def path_waypoints(path, spacing=constants.ARTIFACT_WAYPOINT_SPACING,
                   step=None):
    """Resample an RRT path to (M, 2) world waypoints.

    Walks every edge's arc and emits points at most `spacing` meters
    apart (plus the exact path states), so a tracking controller can
    treat consecutive waypoints as straight segments. The optional
    run_rrt.py --save-artifacts flag writes these arrays; PS1 does not
    require them, and PS2 supplies its own reference trajectory.
    """
    if step is None:
        step = constants.RRT_STEP
    pts = [np.asarray(path[0], dtype=float)[:2]]
    n_sub = max(1, int(math.ceil(step / spacing)))
    for parent, child in zip(path, path[1:]):
        pts.extend(edge_arc(parent, child, step, n_points=n_sub)[1:])
    return np.array(pts)
