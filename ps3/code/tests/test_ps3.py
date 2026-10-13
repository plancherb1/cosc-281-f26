"""Released public tests; staff-only checks are omitted."""
import numpy as np

import pytest

import gradelib

import staff_common

def _pf_module():
    return gradelib.load("particle_filter")

def _tiny_filter(n, rng_seed=0):
    """A filter on the corridor with the staff models, sized n. The
    resampling tests overwrite particles and weights immediately; the
    constructor only has to run."""
    models = staff_common.staff_models()
    world = staff_common.corridor()
    beam_angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    return _pf_module().ParticleFilter(
        world, n, np.random.default_rng(rng_seed),
        models.MotionModel(),
        models.SensorModel(world, beam_angles, 5.0),
        init_pose=world.start)

@pytest.mark.public
def test_resample_bitwise_p3_case():
    """A fixed systematic-resampling boundary example.
    weights (0.1, 0.3, 0.1, 0.4, 0.1), u0 = 0.10:
      cdf   = (0.1, 0.4, 0.5, 0.9, 1.0)
      u     = (0.1, 0.3, 0.5, 0.7, 0.9)
      picks : cdf_0 = 0.1 is NOT > 0.1 -> 1; 0.3 -> 1; cdf_2 = 0.5 is
              NOT > 0.5 -> 3; 0.7 -> 3; cdf_3 = 0.9 is NOT > 0.9 -> 4
    Selected indices [1, 1, 3, 3, 4] (0-based). A >= implementation
    selects [0, 1, 2, 3, 3] instead: three of five picks sit exactly on
    cdf ties, by construction. All the pinned comparisons are exact in
    float64 (every cumsum partial and every u_i rounds to the same
    binary double the hand numbers denote; margins elsewhere are >=
    0.1 >> 1 ulp). Comparison: np.array_equal, zero tolerance."""
    pf = _tiny_filter(5)
    pf.particles = np.column_stack(
        [np.arange(5.0), np.zeros(5), np.zeros(5)])
    pf.weights = np.array([0.1, 0.3, 0.1, 0.4, 0.1])
    pf.resample(u0=0.10)
    assert np.array_equal(pf.particles[:, 0],
                          np.array([1.0, 1.0, 3.0, 3.0, 4.0])), \
        "systematic resampling picks differ from the course rule " \
        "(first j with cdf_j STRICTLY > u_i, u_i = u0 + i/N, in order)"
    assert np.array_equal(pf.weights, np.full(5, 1.0 / 5.0)), \
        "weights must be reset to exactly 1/N after resampling"

@pytest.mark.public
def test_resample_bitwise_n8_offgrid_u0():
    """Second vector, N = 8 with an off-grid u0, to catch hardcoded
    N = 5 or u0-ignoring implementations.
    weights (0.05, 0.05, 0.1, 0.2, 0.05, 0.25, 0.1, 0.2), u0 = 0.03:
      cdf = (0.05, 0.10, 0.20, 0.40, 0.45, 0.70, 0.80, 1.00)
      u   = (0.03, 0.155, 0.28, 0.405, 0.53, 0.655, 0.78, 0.905)
      picks 0, 2, 3, 4, 5, 5, 6, 7  (every margin >= 0.02 >> 1 ulp,
      so no float knife edges). Hand-derived from the course rule."""
    pf = _tiny_filter(8)
    pf.particles = np.column_stack(
        [np.arange(8.0), np.zeros(8), np.zeros(8)])
    pf.weights = np.array([0.05, 0.05, 0.1, 0.2, 0.05, 0.25, 0.1, 0.2])
    pf.resample(u0=0.03)
    assert np.array_equal(pf.particles[:, 0],
                          np.array([0.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0,
                                    7.0]))
    assert np.array_equal(pf.weights, np.full(8, 1.0 / 8.0)), \
        "weights must be reset to exactly 1/N after resampling"

@pytest.mark.public
def test_correct_small_case():
    """One small hand-followable correct() case (public tier): 4 fixed
    particles, an 8-beam synthetic corridor scan, non-uniform prior.
    Expected weights follow the course formula (multiply prior by
    exp(log_lik - max), renormalize) through the SAME staff sensor
    model the filter holds, so this isolates the correct() plumbing.
    atol 1e-9 covers summation-order slack only."""
    models = staff_common.staff_models()
    world = staff_common.corridor()
    beam_angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    sm = models.SensorModel(world, beam_angles, 5.0)
    pf = _pf_module().ParticleFilter(
        world, 4, np.random.default_rng(3), models.MotionModel(), sm,
        init_pose=world.start)
    pf.particles = np.array([[1.5, 1.5, 0.0],
                             [1.6, 1.4, 0.1],
                             [3.0, 1.5, 0.0],
                             [9.0, 1.8, 1.0]])
    pf.weights = np.array([0.4, 0.3, 0.2, 0.1])
    # A plausible corridor scan: back/side walls ~1.3-1.8 m, front open.
    scan = np.array([1.3, 1.8, 1.3, 1.8, 5.0, 1.8, 1.3, 1.8])

    log_lik = sm.log_likelihood(pf.particles.copy(), scan)
    expected = pf.weights * np.exp(log_lik - log_lik.max())
    expected = expected / expected.sum()
    before = pf.particles.copy()

    pf.correct(scan)
    assert np.array_equal(pf.particles, before), \
        "correct() must not move particles"
    assert np.allclose(pf.weights, expected, rtol=0, atol=1e-9), \
        "weights after correct() differ from prior x likelihood, " \
        "renormalized (did you overwrite instead of multiply?)"
    assert abs(pf.weights.sum() - 1.0) < 1e-12

@pytest.mark.public
@pytest.mark.parametrize("case", ["pid_overshoot", "pid_slow", "pid_reject",
                                 "lqr_responsive", "lqr_gentle"])
def test_control_response(case):
    experiment = staff_common.control_experiment()
    config = gradelib.load("control_gains")
    results = experiment.evaluate(config)
    checks = experiment.acceptance(results)
    groups = {
        "pid_overshoot": ["pid_overshoot_bounded", "pid_visible_overshoot",
                          "pid_overshoot_settles"],
        "pid_slow": ["pid_slow_bounded", "pid_slow_shape"],
        "pid_reject": ["pid_reject_bounded", "pid_rejects_load"],
        "lqr_responsive": ["lqr_responsive_bounded", "lqr_responsive_shape"],
        "lqr_gentle": ["lqr_gentle_bounded", "lqr_gentle_shape"],
    }
    failed = [name for name in groups[case] if not checks[name]]
    assert not failed, (
        f"{case}: adjust {', '.join(failed)}. "
        f"Measured response: {experiment.metrics(results[case])}"
    )

