"""PS1 Task 2: policy iteration. Implement improve_policy only.

The fixed-policy evaluation and outer iteration are provided. The model
is known, just as in value iteration; no episodes or learning rate are used.

compare.py calls policy_iteration(), which alternates evaluate_policy()
and improve_policy() until the action grid is unchanged. Complete TODO I
by reusing greedy policy extraction from vi.py. The supplied V contains
the value of the current policy at every free cell; the linear system
used to compute these values is provided in evaluate_policy().
"""
import numpy as np

from vi import extract_policy


def evaluate_policy(world, policy, gamma):
    """Evaluate a fixed policy by solving V = r_pi + gamma P_pi V.

    Return a grid-shaped value array, with NaN on walls and zero at the
    goal. For 0 <= gamma < 1, the system is nonsingular even when the
    policy contains a cycle. This function is provided.
    """
    if not 0 <= gamma < 1:
        raise ValueError("policy evaluation requires 0 <= gamma < 1")
    cells = [s for s in world.free_cells() if s != world.goal]
    indices = {s: i for i, s in enumerate(cells)}
    matrix = np.eye(len(cells))
    rewards = np.zeros(len(cells))
    for i, s in enumerate(cells):
        a = int(policy[s])
        if not 0 <= a < 4:
            raise ValueError("nonterminal policy actions must be in 0..3")
        s2, reward, done = world.step(s, a)
        rewards[i] = reward
        if not done:
            matrix[i, indices[s2]] -= gamma
    values = np.linalg.solve(matrix, rewards) if cells else np.empty(0)
    V = np.full(world.shape, np.nan)
    V[world.goal] = 0.0
    for s, value in zip(cells, values):
        V[s] = value
    return V


def improve_policy(world, V, gamma):
    """Return an integer action grid greedy with respect to V.

    For each free nonterminal cell, maximize r + gamma * V[s2] using
    world.step(s, a); omit the future term when done. Break exact ties
    toward the lower action index (N,S,E,W). Return -1 on walls and goal.
    Do not change V. This is the same lookahead as vi.extract_policy.
    """
    # TODO I: reuse your extract_policy(world, V, gamma) from vi.py and
    # return its action grid. This requires a single function call:
    # policy improvement and greedy policy extraction are the same step.
    # Use the supplied current-policy values, not a new value-iteration run.
    raise NotImplementedError("TODO I: reuse greedy policy extraction")


def policy_iteration(world, gamma, max_rounds=1000):
    """Alternate policy evaluation and improvement, starting with all N.

    Returns (V, policy, rounds), including the final unchanged round.
    This function is provided. It raises RuntimeError if max_rounds is
    reached before the policy stabilizes.
    """
    policy = np.full(world.shape, -1, dtype=int)
    for s in world.free_cells():
        if s != world.goal:
            policy[s] = 0
    for rounds in range(1, max_rounds + 1):
        V = evaluate_policy(world, policy, gamma)
        improved = improve_policy(world, V, gamma)
        if np.array_equal(improved, policy):
            return V, policy, rounds
        policy = improved
    raise RuntimeError("policy iteration did not stabilize before max_rounds")
