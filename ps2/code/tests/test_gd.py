"""Released public tests; staff-only checks are omitted."""
import os

import re

import numpy as np

import pytest

import gradelib

import config

def _gd_mod():
    return gradelib.load("gd")

@pytest.mark.public
def test_grad_matches_fd():
    """mse_grad vs central finite differences at 3 random thetas."""
    mod = _gd_mod()
    x, y = mod.make_curve_data()
    X = mod.polyfeatures(x)
    rng = np.random.default_rng(config.SEED)
    eps = 1e-6
    for _ in range(3):
        theta = rng.normal(0.0, 1.0, X.shape[1])
        analytic = np.asarray(mod.mse_grad(theta, X, y), dtype=float)
        fd = np.empty_like(analytic)
        for i in range(len(theta)):
            e = np.zeros_like(theta)
            e[i] = eps
            fd[i] = (mod.mse_loss(theta + e, X, y)
                     - mod.mse_loss(theta - e, X, y)) / (2 * eps)
        np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-8)

@pytest.mark.public
def test_gd_quadratic_exact():
    """gd on f(theta) = ||theta - c||^2 with alpha 0.25 contracts the
    gap by exactly half per step: 100 iters lands within 1e-6 of c.
    Catches sign and step-size bugs in closed form."""
    mod = _gd_mod()
    c = np.array([2.0, -1.0, 0.5])
    grad_fn = lambda th: 2.0 * (th - c)
    loss_fn = lambda th: float(np.sum((th - c) ** 2))
    theta, losses = mod.gd(np.zeros(3), grad_fn, loss_fn, 0.25, 100)
    assert np.allclose(theta, c, atol=1e-6), \
        f"gd landed at {theta}, expected {c} (sign flip walks away)"
    assert len(losses) == 101, \
        "losses must have length iters + 1 (loss before each update, " \
        "plus the final loss)"
    assert losses[0] == pytest.approx(loss_fn(np.zeros(3)))
    assert losses[-1] < 1e-10

@pytest.mark.public
def test_gd_monotone_smallstep():
    """At the smallest sweep learning rate the loss never increases."""
    mod = _gd_mod()
    x, y = mod.make_curve_data()
    X = mod.polyfeatures(x)
    _, losses = mod.gd(np.zeros(X.shape[1]),
                       lambda th: mod.mse_grad(th, X, y),
                       lambda th: mod.mse_loss(th, X, y),
                       config.LR_SWEEP[0], config.GD_ITERS)
    diffs = np.diff(np.asarray(losses, dtype=float))
    assert (diffs <= 1e-12).all(), \
        "loss increased at the timid learning rate"

_ALLOWED_RNG = re.compile(r"np\.random\.(default_rng|Generator)")

@pytest.mark.public
def test_no_global_rng():
    """Course CI rule (SIM_SPEC requirement 5): no global np.random
    use in the submitted files; all randomness through a Generator."""
    for name in ("gd",):
        path = os.path.join(gradelib.submission_dir(), name + ".py")
        with open(path) as f:
            source = f.read()
        for match in re.finditer(r"np\.random\.\w+", source):
            assert _ALLOWED_RNG.match(match.group()), \
                f"{name}.py uses global {match.group()}; draw from a " \
                "np.random.default_rng(...) Generator instead"

