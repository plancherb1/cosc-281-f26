"""PS2 pinned constants. PROVIDED, DO NOT EDIT.

This is the single source for values shared by the PS2 gradient-descent scripts and the public autograder.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 3701                      # the one PS2 seed (data, sim, shuffles)

# ------------------------------------------------- Task: GD from scratch
CURVE_N = 40                     # dataset size
CURVE_NOISE_STD = 0.1            # observation noise, y units
LR_SWEEP = [2e-4, 1e-1, 3.397e-1, 5e-1]   # timid -> reckless, 4 regimes
GD_ITERS = 200                   # iterations per sweep run
