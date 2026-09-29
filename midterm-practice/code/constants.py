"""Grid-learning experiment settings. PROVIDED, DO NOT EDIT."""
CELL_SIZE_M = 1.0
GOAL_REWARD = 10.0
GAMMA = 0.9

# --------------------------------------------------- Q-learning, Task 1
Q_ALPHA = 0.5            # learning rate
Q_SEED = 26              # rng seed for the Task 1 runs
# Per-map training budgets: (episodes, epsilon, max_steps). narrow needs
# the larger budget; see the header note.
Q_BUDGET = {
    "open":   {"episodes": 2000, "epsilon": 0.4, "max_steps": 200},
    "maze":   {"episodes": 2000, "epsilon": 0.4, "max_steps": 200},
    "narrow": {"episodes": 4000, "epsilon": 0.8, "max_steps": 500},
}
