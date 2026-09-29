"""Run Midterm practice passive evaluation and Q-learning experiments. PROVIDED.

Run python midterm-practice/code/run_learning.py from the course root. The passive
experiment follows a fixed stochastic policy on a 2x2 GridMDP: choose
uniformly among actions that move to a different cell. The terminal
state is the upper-right cell. All episodes are completed, without
truncation. Exact values below are computed only for plotting/evaluation.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents
            if (parent / 'sim' / 'coursesim').is_dir())
try:
    import coursesim
except ImportError:
    sys.path.insert(0, str(ROOT / 'sim'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import constants
from grid_mdp import GridMDP, greedy_rollout
from passive_rl import direct_evaluation, td_evaluation
from q_learning import q_learning


def passive_data(count=1000):
    world = GridMDP(np.zeros((2, 2), dtype=bool), (1, 0), (0, 1))
    cells = [(1, 0), (0, 0), (1, 1), (0, 1)]
    index = {s: i for i, s in enumerate(cells)}
    choices = {s: [a for a in range(4) if world.step(s, a)[0] != s]
               for s in cells[:-1]}
    matrix, rewards = np.eye(3), np.zeros(3)
    for s in cells[:-1]:
        for a in choices[s]:
            s2, reward, done = world.step(s, a)
            prob = 1 / len(choices[s])
            rewards[index[s]] += prob * reward
            if not done:
                matrix[index[s], index[s2]] -= prob * constants.GAMMA
    exact = np.append(np.linalg.solve(matrix, rewards), 0.)
    rng = np.random.default_rng(281)
    episodes = []
    for _ in range(count):
        episode, s = [], world.start
        while s != world.goal:
            a = int(rng.choice(choices[s]))
            s2, reward, done = world.step(s, a)
            episode.append((index[s], reward, index[s2], done))
            s = s2
        episodes.append(episode)
    return episodes, exact


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    episodes, exact = passive_data()
    direct = direct_evaluation(episodes, 4, constants.GAMMA)
    td = td_evaluation(episodes, 4, constants.GAMMA, .1)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    for offset, vals, label in [(-.25, exact, 'Exact fixed-policy value'),
                               (0, direct, 'Direct evaluation'),
                               (.25, td, 'TD, alpha=0.1')]:
        ax.bar(x+offset, vals, .25, label=label)
    ax.set_xticks(x, ['(1,0)', '(0,0)', '(1,1)', 'Goal'])
    ax.set(xlabel='Grid state (row, column)', ylabel='Value',
           title='1000 completed episodes of the same fixed policy')
    ax.set_ylim(0, 1.4 * max(np.max(exact), np.max(direct), np.max(td)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / 'practice_learning.png', dpi=150)
    plt.close(fig)
    print('Passive values (start, upper-left, lower-right, goal):')
    print('exact ', exact, '\ndirect', direct, '\nTD    ', td)
    print(f"{'map':<10}{'epsilon':>9}{'successes/episodes':>22}{'updates':>12}{'greedy steps':>15}")
    for name, no_explore in [('open', False), ('maze', False),
                             ('narrow', False), ('maze', True)]:
        world = GridMDP.from_map(name)
        budget = constants.Q_BUDGET[name]
        epsilon = 0. if no_explore else budget['epsilon']
        Q, stats = q_learning(world, budget['episodes'], constants.Q_ALPHA,
                              constants.GAMMA, epsilon, budget['max_steps'],
                              np.random.default_rng(constants.Q_SEED))
        steps = greedy_rollout(world, Q, max_steps=1000)
        successes = f'{stats.successes}/{stats.episodes}'
        print(f"{name:<10}{epsilon:>9.1f}{successes:>22}{stats.updates:>12}{str(steps):>15}")
    print('None means the greedy rollout did not reach the goal.')


if __name__ == '__main__':
    main()
