"""PS1 Task 2: value iteration and greedy policy extraction.

compare.py builds a GridMDP, calls value_iteration(), then extract_policy(),
and plots the resulting values and actions. Complete TODO V and TODO P.
Initialization, state loops, convergence checking, and return values are
provided. You do not need to load maps or implement the transition model.

The world interface used here is:
  world.shape: (number of rows, number of columns).
  world.free_cells(): list of (row, col) tuples, including the goal.
  world.goal: terminal (row, col) tuple.
  world.step(s, a): returns (next_cell, reward, done).
Actions are integer indices 0=N, 1=S, 2=E, 3=W. A blocked move stays
in place. Reward is +10 on entering the goal, zero otherwise. done is
True when the next cell is the goal. Use the gamma argument (the runner
passes 0.9); do not hard-code it. V[s] indexes the array at tuple s.
"""
import numpy as np

from grid_mdp import ACTIONS


def value_iteration(world, gamma, tol, max_sweeps=1000):
    """Compute the optimal value function using synchronous value iteration.

    Args:
        world: A GridMDP with the interface described above.
        gamma: The discount factor.
        tol: The convergence threshold for the largest per-cell change.
        max_sweeps: The maximum number of complete sweeps.

    Returns:
        (V, sweeps_run), where V has world.shape, walls are NaN, and the
        terminal goal has value zero. The count includes the final sweep.

    The update and stopping conventions are:
      * Synchronous sweeps: every backup in sweep k reads the value
        table produced by sweep k-1 (update a copy, then swap).
      * V0 = 0 on every free cell. Walls retain NaN. The goal is
        terminal with V[goal] = 0 and is never backed up.
      * The backup at a free cell s is
            V(s) = max_a [ r(s, a) + gamma * V(s') ]
        with (s', r, done) from world.step(s, a), and the future term
        dropped when done (the episode ends at the goal).
      * Stop after the first sweep whose largest per-cell change is
        below tol; return that sweep's table and the number of sweeps
        run (that final sweep included).
    """
    # Initialize free cells to zero and mark walls with NaN.
    cells = world.free_cells()
    V = np.full(world.shape, np.nan)
    for s in cells:
        V[s] = 0.0

    sweeps_run = 0
    for sweeps_run in range(1, max_sweeps + 1):
        V_new = V.copy()
        largest_change = 0.0
        for s in cells:
            if s == world.goal:
                continue
            # TODO V: write the Bellman backup for this one state into
            # V_new[s]. For each integer a in range(len(ACTIONS)), call
            # world.step(s, a). Its one-step return is reward if done,
            # otherwise reward + gamma * V[s2]. Take the largest return.
            # Read successor values from V, never V_new (synchronous VI).
            raise NotImplementedError("TODO V: back up one state")
            largest_change = max(largest_change, abs(V_new[s] - V[s]))
        V = V_new
        if largest_change < tol:
            break
    return V, sweeps_run


def extract_policy(world, V, gamma):
    """Extract a policy that is greedy with respect to the supplied V.

    Returns an (H, W) int array: for each free non-goal cell the argmax
    action of the one-step lookahead r + gamma * V(s') (future term
    dropped when the step enters the goal); -1 on walls and on the goal
    cell. Break exact ties toward the lower action index in the order
    N, S, E, W. Applying np.argmax to returns in this order selects the
    required action. Leave V unchanged and use the supplied gamma.
    """
    # Use -1 for walls and the terminal goal, where no action is required.
    policy = np.full(world.shape, -1, dtype=int)
    for s in world.free_cells():
        if s == world.goal:
            continue
        # TODO P: evaluate the same four one-step returns as TODO V,
        # using the supplied V without changing it. Store the index of
        # the maximizing action in policy[s], not its return value.
        # Evaluate actions in order 0, 1, 2, 3; np.argmax then chooses
        # the first maximum and gives the required tie-breaking rule.
        raise NotImplementedError("TODO P: choose one greedy action")
    return policy
