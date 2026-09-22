"""Released public tests; staff-only checks are omitted."""
import math

import numpy as np

import pytest

import gradelib

from coursesim import CarParams, Map

import constants

from dubins import DubinsWorld, _primitive_end

def _rrt_fn():
    return gradelib.load("rrt").rrt

def _run(world, seed=constants.RRT_SEED):
    return _rrt_fn()(world, world.start, world.goal,
                     np.random.default_rng(seed), constants.RRT_STEP,
                     constants.RRT_GOAL_BIAS, constants.RRT_GOAL_TOL,
                     constants.RRT_MAX_ITERS)

def _check_result_contract(world, result, expect_path):
    assert len(result.tree) >= 1
    root, root_parent = result.tree[0]
    assert np.allclose(root, world.start) and root_parent == -1
    assert result.nodes_added == len(result.tree) - 1, \
        "nodes_added counts accepted extensions, excluding the root. Leave the scaffold's insertion and counting code unchanged."
    assert 0 < result.samples_drawn <= constants.RRT_MAX_ITERS
    for state, parent in result.tree[1:]:
        assert 0 <= parent < len(result.tree)
        assert world.collision_free(state)
    if expect_path:
        assert result.path is not None, (
            "No path found within the budget. Check TODO N: compare every "
            "tree node's squared (x,y) distance to the sample, ignoring theta. "
            "In TODO E, pass the nearest node's state, not its parent index, "
            "to dubins_steer. Keep the provided random draws unchanged.")
        assert np.allclose(result.path[0], world.start)
        end = result.path[-1]
        assert math.hypot(end[0] - world.goal[0], end[1] - world.goal[1]) \
            < constants.RRT_GOAL_TOL
        for a, b in zip(result.path, result.path[1:]):
            assert _is_primitive_edge(a, b), \
                "A path edge is not a car motion. TODO E must call dubins_steer from the selected node; do not connect directly to the sample."

def _is_primitive_edge(a, b, step=constants.RRT_STEP):
    car = CarParams()
    for steer in (car.max_steer, 0.0, -car.max_steer):
        end = _primitive_end(a, steer, step)
        if math.hypot(end[0] - b[0], end[1] - b[1]) < 1e-9 \
                and abs(end[2] - b[2]) < 1e-9:
            return True
    return False

def _empty_world():
    """12x12 m empty room (walled border), in-memory, never a course
    map. 0.05 m resolution to match the course convention."""
    occ = np.zeros((240, 240), dtype=bool)
    occ[:3, :] = occ[-3:, :] = True
    occ[:, :3] = occ[:, -3:] = True
    m = Map(occ, 0.05, start=(2.0, 2.0, 0.0), goal=(10.0, 10.0, 0.0))
    return DubinsWorld(m, margin=0.05)

def _wall_world():
    """One wall band with a wide doorway; forces a detour."""
    occ = np.zeros((240, 240), dtype=bool)
    occ[:3, :] = occ[-3:, :] = True
    occ[:, :3] = occ[:, -3:] = True
    occ[100:110, :160] = True          # wall at y ~ 6.5-7.0, door x > 8
    m = Map(occ, 0.05, start=(2.0, 2.0, 0.0), goal=(2.0, 10.0, 1.57))
    return DubinsWorld(m, margin=0.05)

@pytest.mark.public
def test_rrt_empty_world():
    world = _empty_world()
    result = _run(world, seed=3)
    _check_result_contract(world, result, expect_path=True)

@pytest.mark.public
def test_rrt_wall_world():
    world = _wall_world()
    result = _run(world, seed=3)
    _check_result_contract(world, result, expect_path=True)

@pytest.mark.public
def test_rrt_deterministic():
    world = _empty_world()
    first = _run(world, seed=11)
    second = _run(world, seed=11)
    assert first.samples_drawn == second.samples_drawn
    assert first.nodes_added == second.nodes_added
    a = np.array([s for s, _ in first.tree])
    b = np.array([s for s, _ in second.tree])
    np.testing.assert_array_equal(a, b, err_msg=
        "Two runs with the same seed produced different trees. Use only the supplied RNG and preserve sampling order; TODO N must keep the earliest node on ties.")

