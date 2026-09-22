"""Shared experimental parameters for PS1.

The runners and autograder use these values to make results comparable.
Use the supplied parameters for the required experiments. No changes to
this file are required.

Units: meters for lengths, radians for angles, unitless otherwise.
"""

# ----------------------------------------------------------- grid MDP
CELL_SIZE_M = 1.0        # MDP cell size, m (20 x the 0.05 m/px maps)
GOAL_REWARD = 10.0       # reward on entering the terminal goal cell
GAMMA = 0.9              # discount factor

# ------------------------------------------------------ value iteration
VI_TOL = 1e-8            # max per-cell change that counts as converged
VI_MAX_SWEEPS = 1000     # maximum number of complete sweeps

# --------------------------------------------------------- RRT, Task 3
RRT_STEP = 0.3           # arc length of one motion primitive, m
RRT_GOAL_BIAS = 0.05     # probability a sample is the goal itself
RRT_GOAL_TOL = 0.5       # termination radius around the goal position, m
RRT_MAX_ITERS = 8000     # iteration budget per map
RRT_SEED = 26            # rng seed for every RRT run
# Planning clearance added to the car footprint radius, per map (m).
RRT_MARGIN = {"open": 0.05, "maze": 0.05, "narrow": 0.0}
# Goal pose [x, y, theta] per map; None = the map's own goal. See the
# run_rrt.py explanation of why maze plans to a checkpoint.
RRT_GOALS = {"open": None, "maze": (13.5, 7.5, 0.0), "narrow": None}

# ------------------------------------------------------------ artifacts
# compare.py writes out/values_<map>.npy; run_rrt.py --save-artifacts
# writes out/rrt_path_<map>.npy ((M, 2) world waypoints, spacing <=
# ARTIFACT_WAYPOINT_SPACING). These optional arrays are not submitted.
ARTIFACT_DIR = "out"
ARTIFACT_WAYPOINT_SPACING = 0.25   # m between saved path waypoints
