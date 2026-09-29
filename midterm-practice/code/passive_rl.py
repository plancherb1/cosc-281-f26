"""Midterm practice Task 1: passive evaluation of a fixed policy. Implement both functions.

Episodes are lists of (state, reward, next_state, done) tuples. States
are integer indices in 0..n_states-1; the terminal state has index n_states-1.
Every supplied episode ends with done=True. Do not modify the episodes.
"""
import numpy as np


def direct_evaluation(episodes, n_states, gamma):
    """Every-visit average of discounted returns, as a float array.

    Compute the return from each visited state to the end of its episode.
    Count repeated visits separately. Return zero for unvisited states
    and for the terminal state. Do not treat truncated trajectories as
    completed episodes; the runner supplies only completed episodes.
    """
    raise NotImplementedError("implement direct_evaluation()")


def td_evaluation(episodes, n_states, gamma, alpha):
    """TD(0), zero initialization, one pass in the supplied episode order.

    Process each episode from first transition to last. Set sample=r
    if done, otherwise sample=r+gamma*V[next_state]. Update only V[state]
    by V[state] += alpha*(sample-V[state]). Values persist across episodes.
    Keep the terminal value zero. Return the final float value array.
    """
    raise NotImplementedError("implement td_evaluation()")
