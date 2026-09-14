"""Seeded random-stream tree for the simulator (course simulator specification section `rng.py`).

One `np.random.SeedSequence` per Simulator episode. Every stochastic
component draws from its own child Generator, so the same seed replays
bit-for-bit no matter which sensors a given assignment reads.

The spawn order below is FROZEN. Children are spawned in one batch, in
STREAMS order, at every reset. `camera` is reserved now, before any camera
code exists, so adding the camera later does not shift the lidar, odometry,
or initial-pose streams of already-graded seeds.

Draw-order contract (who consumes what, per Simulator call):
  reset(): init_pose draws normal(size=3); lidar draws normal(size=n_beams)
           then uniform(size=n_beams). Odometry draws nothing at reset.
  step():  odometry draws normal(size=3); lidar draws normal(size=n_beams)
           then uniform(size=n_beams).
Sensors always make their draws, even when their noise scales are zero, so
turning a noise knob never shifts any stream.

Global `np.random` calls (np.random.seed, np.random.normal, ...) are BANNED
everywhere under sim/; `sim/tests/test_no_global_random.py` enforces this
by grep. Constructing generators via `np.random.default_rng`,
`np.random.Generator`, `np.random.PCG64`, or `np.random.SeedSequence` is
allowed; drawing from the global state is not.
"""
import numpy as np

# Frozen spawn order. Index in this tuple == child index off the root
# SeedSequence. Append only; never reorder or remove.
STREAMS = ("lidar", "odometry", "init_pose", "camera")


def make_streams(seed):
    """Build the per-component generator dict for one episode.

    seed: int (or any SeedSequence-compatible entropy). Returns
    dict[str, np.random.Generator] keyed by STREAMS. Also returns the root
    SeedSequence so callers can spawn extra children (e.g. a particle
    filter) that descend from the same seed without touching the four
    fixed component streams.

    Returns (streams, root) where streams is the dict and root is the
    np.random.SeedSequence.
    """
    root = np.random.SeedSequence(seed)
    children = root.spawn(len(STREAMS))
    streams = {name: np.random.Generator(np.random.PCG64(child))
               for name, child in zip(STREAMS, children)}
    return streams, root
