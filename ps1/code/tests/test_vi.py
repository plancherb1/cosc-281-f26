"""Released public tests; staff-only checks are omitted."""
from collections import deque

import numpy as np

import pytest

import gradelib

import constants

from grid_mdp import ACTIONS, GridMDP, greedy_rollout

def _vi_mod():
    return gradelib.load("vi")

def bfs_distances(world):
    """Shortest obstacle-respecting distance to the goal per free cell."""
    dist = {world.goal: 0}
    queue = deque([world.goal])
    n_rows, n_cols = world.shape
    while queue:
        row, col = queue.popleft()
        for drow, dcol in ACTIONS:
            nb = (row + drow, col + dcol)
            if 0 <= nb[0] < n_rows and 0 <= nb[1] < n_cols \
                    and not world.occupancy[nb] and nb not in dist:
                dist[nb] = dist[(row, col)] + 1
                queue.append(nb)
    return dist

def oracle_table(world, gamma=constants.GAMMA):
    dist = bfs_distances(world)
    V = np.full(world.shape, np.nan)
    for s in world.free_cells():
        V[s] = 0.0 if s == world.goal \
            else constants.GOAL_REWARD * gamma ** (dist[s] - 1)
    return V, dist

def check_table(world, V, sweeps):
    ref, dist = oracle_table(world)
    _check_values(V, ref)
    assert V[world.goal] == 0.0
    d_max = max(dist.values())
    assert abs(sweeps - (d_max + 1)) <= 1, \
        f"expected about {d_max + 1} synchronous sweeps, got {sweeps} " \
        "(asynchronous or in-place updates converge in fewer)"

def _check_values(V, ref):
    """Point to one incorrect cell while retaining the original tolerance."""
    V = np.asarray(V)
    assert V.shape == ref.shape, f"Value-table shape: expected {ref.shape}, got {V.shape}. Return the full grid, including walls."
    matches = np.isclose(V, ref, rtol=1e-7, atol=1e-6, equal_nan=True)
    if not matches.all():
        cell = tuple(int(x) for x in np.argwhere(~matches)[0])
        raise AssertionError(
            f"VI value at cell {cell}: expected {ref[cell]:.8g}, got {V[cell]:.8g}. "
            "Check TODO V: maximize over all four action returns, use the "
            "supplied gamma, omit future reward on termination, and read V "
            "rather than V_new. Walls must remain NaN and the goal zero.")

SYN = ["......",
       "..##..",
       "...#..",
       "......"]

SYN_START, SYN_GOAL = (3, 0), (0, 5)

def _syn_world():
    occ = np.array([[ch == "#" for ch in row] for row in SYN])
    return GridMDP(occ, SYN_START, SYN_GOAL, cell_size=1.0)

@pytest.mark.public
def test_vi_synthetic_values_exact():
    world = _syn_world()
    V, sweeps = _vi_mod().value_iteration(
        world, constants.GAMMA, constants.VI_TOL, constants.VI_MAX_SWEEPS)
    check_table(world, V, sweeps)

@pytest.mark.public
def test_vi_synthetic_policy_rollout():
    world = _syn_world()
    vi = _vi_mod()
    V, _ = vi.value_iteration(world, constants.GAMMA, constants.VI_TOL)
    policy = vi.extract_policy(world, V, constants.GAMMA)
    _, dist = oracle_table(world)
    steps = greedy_rollout(world, policy, max_steps=50)
    assert steps == dist[world.start], (
        f"Greedy policy took {steps} steps (None means stuck or over budget); "
        f"expected {dist[world.start]}. Check TODO P: store an action INDEX "
        "0=N, 1=S, 2=E, 3=W, not the action's value.")

@pytest.mark.public
def test_vi_walls_stay_nan():
    world = _syn_world()
    V, _ = _vi_mod().value_iteration(world, constants.GAMMA,
                                     constants.VI_TOL)
    assert np.isnan(V[world.occupancy]).all(), "Wall entries must remain NaN; update only the free nonterminal states in the scaffold."
    assert np.isfinite(V[~world.occupancy]).all(), "A free cell has a NaN or infinite value. Use world.step for transitions rather than indexing through walls."

