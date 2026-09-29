"""Released public tests; staff-only checks are omitted."""
import copy

import numpy as np

import pytest

import gradelib

@pytest.mark.public
def test_written_episodes():
    episodes = [[(0, 0., 1, False), (1, 4., 2, True)],
                [(0, 0., 1, False), (1, 2., 2, True)]]
    original = copy.deepcopy(episodes)
    mod = gradelib.load('passive_rl')
    np.testing.assert_allclose(mod.direct_evaluation(episodes, 3, .5), [1.5, 3, 0])
    np.testing.assert_allclose(mod.td_evaluation(episodes, 3, .5, .5), [.5, 2, 0])
    assert episodes == original

@pytest.mark.public
def test_every_visit_and_unvisited():
    episodes = [[(0, 1., 0, False), (0, 4., 3, True)]]
    mod = gradelib.load('passive_rl')
    np.testing.assert_allclose(mod.direct_evaluation(episodes, 4, .5), [3.5, 0, 0, 0])
    np.testing.assert_allclose(mod.td_evaluation(episodes, 4, .5, .5), [2.25, 0, 0, 0])

