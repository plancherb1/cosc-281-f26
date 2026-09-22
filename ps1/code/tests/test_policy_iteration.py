"""Released public tests; staff-only checks are omitted."""
import numpy as np

import pytest

import gradelib

from grid_mdp import GridMDP, greedy_rollout

@pytest.mark.public
def test_policy_improvement():
    world = GridMDP(np.zeros((2, 2), dtype=bool), (1, 0), (0, 1))
    V = np.array([[10., 0.], [9., 10.]])
    result = gradelib.load('policy_iteration').improve_policy(world, V, .9)
    np.testing.assert_array_equal(result, [[2, -1], [0, 0]], err_msg=
        "Check TODO P/I: return action indices (0=N,1=S,2=E,3=W), choose the lower index on ties, and use -1 at the goal. TODO I should reuse extract_policy.")
    np.testing.assert_array_equal(V, [[10, 0], [9, 10]], err_msg=
        "Policy improvement must not modify its input value table. Return a separate action grid.")

@pytest.mark.public
def test_policy_iteration_synthetic():
    world = GridMDP(np.zeros((2, 2), dtype=bool), (1, 0), (0, 1))
    V, policy, rounds = gradelib.load('policy_iteration').policy_iteration(world, .9)
    np.testing.assert_allclose(V, [[10, 0], [9, 10]], err_msg=
        "The provided PI loop did not reach the expected values. Check greedy policy extraction and reuse it in TODO I; leave policy evaluation unchanged.")
    assert greedy_rollout(world, policy) == 2
    assert rounds >= 2

