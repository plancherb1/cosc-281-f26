"""Released public tests; staff-only checks are omitted."""
import numpy as np

import pytest

import gradelib

from grid_mdp import GridMDP, greedy_rollout

def _q_fn():
    return gradelib.load("q_learning").q_learning

SYN_START, SYN_GOAL = (2, 0), (0, 2)

def _syn_world():
    return GridMDP(np.zeros((3, 3), dtype=bool), SYN_START, SYN_GOAL)

def _reference_q(world, episodes, alpha, gamma, epsilon, max_steps, rng):
    """The pinned contract, executed. Mirrors the q_learning docstring
    line for line; kept tiny and synthetic-only so the private staff
    reference never ships."""
    Q = np.zeros((*world.shape, 4))
    for _ in range(episodes):
        s = world.start
        for _ in range(max_steps):
            if rng.uniform() < epsilon:
                a = int(rng.integers(4))
            else:
                a = int(np.argmax(Q[s]))
            s2, r, done = world.step(s, a)
            target = r if done else r + gamma * np.max(Q[s2])
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                break
    return Q

@pytest.mark.public
def test_q_seeded_synthetic_bitwise():
    world = _syn_world()
    args = dict(episodes=40, alpha=0.5, gamma=0.9, epsilon=0.3,
                max_steps=30)
    expected = _reference_q(world, rng=np.random.default_rng(5), **args)
    Q, stats = _q_fn()(world, rng=np.random.default_rng(5), **args)
    assert Q.shape == (3, 3, 4)
    assert np.array_equal(Q, expected), \
        "Q table differs from the pinned contract; check the rng draw " \
        "order (one uniform() per step, integers(4) only when exploring)"
    assert stats.episodes == args["episodes"]
    assert stats.successes > 0

@pytest.mark.public
def test_q_rng_hygiene():
    """Course CI rule: all randomness flows through the passed rng;
    global np.random state must come back untouched."""
    np.random.seed(1234)
    before = np.random.get_state()
    _q_fn()(_syn_world(), episodes=20, alpha=0.5, gamma=0.9, epsilon=0.5,
            max_steps=20, rng=np.random.default_rng(7))
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]

