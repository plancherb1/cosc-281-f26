"""PS0 C1 gate: verify that the shared simulator environment works.

Public tier, all of it: these tests import no student code. They prove
coursesim imports at the required version and that ten seeded steps run
headless with the frozen Obs fields present. Seed is fixed at 0 per the
harness determinism rule; nothing here reads a clock or global RNG.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.public

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_OPEN_YAML = os.path.join(_ROOT, "sim", "maps", "open.yaml")


def test_import_coursesim_min_version():
    import coursesim
    version = tuple(int(v) for v in coursesim.__version__.split("."))
    assert version >= (0, 1, 0)


def test_ten_steps_headless_obs_fields():
    from coursesim import Simulator
    sim = Simulator(_OPEN_YAML, seed=0)
    obs = sim.reset()
    assert obs.t == 0.0
    assert np.allclose(obs.odom, 0.0)  # zero odom at reset, per Obs spec
    for _ in range(10):
        obs = sim.step((0.5, 0.1))
    for field in ("odom", "scan", "dt", "t", "collided"):
        assert hasattr(obs, field), f"Obs is missing frozen field {field}"
    assert obs.odom.shape == (3,)
    assert obs.scan.ndim == 1 and obs.scan.size > 0
    assert obs.dt == pytest.approx(0.1)
    assert obs.t == pytest.approx(1.0)  # 10 steps of dt = 0.1 s
    assert isinstance(obs.collided, (bool, np.bool_))


def test_reset_replays_bit_for_bit():
    from coursesim import Simulator
    sim = Simulator(_OPEN_YAML, seed=0)
    first = [sim.step((1.0, 0.05)).scan.copy() for _ in range(5)]
    sim.reset()
    second = [sim.step((1.0, 0.05)).scan.copy() for _ in range(5)]
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
