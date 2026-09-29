"""Midterm practice Task 1: tabular Q-learning. THIS FILE IS YOURS TO EDIT.

Implement q_learning() below. Use the supplied random generator and
the draw order specified here so that runs can be reproduced.
"""
import numpy as np

from grid_mdp import Stats


def q_learning(world, episodes, alpha, gamma, epsilon, max_steps, rng):
    """Tabular Q-learning, epsilon-greedy. Q shape = (*world.shape, 4).
    ALL randomness through rng (exploration draws AND epsilon tests, in
    the order: uniform() for explore-vs-exploit, then integers(4) if
    exploring). Start state = world.start each episode. Returns (Q, stats).
    Greedy ties broken by action order (N,S,E,W), pinned.

    Contract details the autograder holds you to:
      * Q starts all zero, dtype float64.
      * Every episode starts at world.start and runs until the goal is
        entered or max_steps steps have been taken.
      * Per step, in this exact order:
          1. draw u = rng.uniform(); explore iff u < epsilon
          2. if exploring, a = int(rng.integers(4));
             else a = int(np.argmax(Q[s]))  (ties -> lower index)
          3. (s2, r, done) = world.step(s, a)
          4. sample = r if done else r + gamma * max_a' Q[s2]
             Q[s][a] += alpha * (sample - Q[s][a])
        No other rng draws anywhere.
      * Book-keeping: call stats.record(s) once per step (step 1 time,
        with the state you act from) and stats.end_episode(reached)
        when the episode ends; return (Q, stats).
    """
    Q = np.zeros((*world.shape, 4))
    stats = Stats()
    # ------------------------------------------------------------------
    # TODO: the training loop, exactly as pinned above. It is a short loop.
    # Common trap: drawing integers(4) on NON-exploring steps changes
    # the stream and fails the seeded test.
    # ------------------------------------------------------------------
    raise NotImplementedError("implement q_learning() in q_learning.py")
