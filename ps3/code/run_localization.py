"""PS3 Task 1 runner: particle-filter localization on the corridor.
PROVIDED, DO NOT EDIT. Your code goes in particle_filter.py.

Usage (from this directory, inside the course venv):

    python run_localization.py            -> out/ps3_localization.png

Drives the car straight down the corridor world (seed 478, 500
particles), runs your filter one Bayes cycle per step, and plots the
position error of your estimate versus time. Until you implement the
three methods, the starter exits with NotImplementedError.

run_filter() returns per-step error and update timing. The timing is
diagnostic information; a particle-count sweep is not required.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
try:
    import coursesim  # noqa: F401  (installed or already on the path)
except ImportError:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "sim"))
    import coursesim  # noqa: F401

import numpy as np

from coursesim import Simulator

from models import MotionModel, SensorModel
from particle_filter import ParticleFilter

SEED = 478                # the handout's deliverable seed
N_PARTICLES = 500
N_STEPS = 150             # 15 m at 1 m/s, start to goal
CONTROL = (1.0, 0.0)      # (v m/s, steer rad): straight down the corridor

_SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(
    coursesim.__file__)))
CORRIDOR_MAP = os.path.join(_SIM_DIR, "maps", "corridor.yaml")


def build_filter(sim, n_particles):
    """The course filter wiring: rng spawned off the simulator seed
    (all randomness descends from ONE seed), provided models, Gaussian
    init around the known start pose (the car knows its starting box;
    the corridor's two ends look identical to a lidar, so a global
    uniform init is deliberately out of scope)."""
    rng = sim.spawn_generator()
    motion = MotionModel()
    sensor = SensorModel(sim.map, sim.lidar.beam_angles,
                         sim.lidar.max_range)
    return ParticleFilter(sim.map, n_particles, rng, motion, sensor,
                          init_pose=sim.map.start)


def run_filter(n_particles, seed=SEED, n_steps=N_STEPS):
    """One full localization run. Returns a dict:

      errors        (n_steps,) |estimate - truth| position error, meters
      step_times_s  (n_steps,) wall-clock of predict + correct +
                    resample ONLY (estimate and simulator excluded)
      estimates     (n_steps, 3) filter estimates
      truths        (n_steps, 3) ground-truth poses (plotting/grading)
    """
    sim = Simulator(CORRIDOR_MAP, seed=seed)
    pf = build_filter(sim, n_particles)
    errors = np.empty(n_steps)
    step_times = np.empty(n_steps)
    estimates = np.empty((n_steps, 3))
    truths = np.empty((n_steps, 3))
    for k in range(n_steps):
        obs = sim.step(CONTROL)
        t0 = time.perf_counter()
        pf.predict(obs.odom)
        pf.correct(obs.scan)
        t1 = time.perf_counter()
        est = pf.estimate()
        t2 = time.perf_counter()
        pf.resample()
        t3 = time.perf_counter()
        step_times[k] = (t1 - t0) + (t3 - t2)
        truth = sim.ground_truth_pose
        errors[k] = float(np.hypot(est[0] - truth[0], est[1] - truth[1]))
        estimates[k] = est
        truths[k] = truth
    return {"errors": errors, "step_times_s": step_times,
            "estimates": estimates, "truths": truths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "out"),
                    help="output directory (default: ./out)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run_filter(N_PARTICLES, seed=SEED, n_steps=N_STEPS)
    half = N_STEPS // 2
    mean_late = float(res["errors"][half:].mean())
    print(f"seed {SEED}, N = {N_PARTICLES}: mean position error over the "
          f"last half = {mean_late:.3f} m "
          f"(autograder band: < 0.30 m on held-out seeds)")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(np.arange(N_STEPS) * 0.1, res["errors"], color="tab:blue")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title(f"PF localization, corridor, seed {SEED}, "
                 f"N = {N_PARTICLES}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(args.out, "ps3_localization.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
