"""PS2 Task 1: gradient descent from scratch.

YOU EDIT THIS FILE, in the marked student section only. No autograd, no
optimizer imports: the pinned course environment has none, and imports
beyond numpy (and the provided helpers) fail the autograder.

Run it to produce the Problem 1 deliverables:

    python ps2/code/gd.py

writes ps2_fit.png (data + fitted curve at the best sweep learning
rate) and ps2_loss_curves.png (all four loss curves, one axes, log-y).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

# ======================================================================
# PROVIDED (do not edit below this line until the student section)
# ======================================================================


def make_curve_data(n=config.CURVE_N, noise_std=config.CURVE_NOISE_STD,
                    seed=config.SEED):
    """The Problem 1 dataset: n points on a smooth NON-polynomial curve.

    x ~ uniform[-1.5, 1.5], y = sin(2x) + Gaussian noise (std
    noise_std). All randomness through np.random.default_rng(seed); the
    global np.random is never touched. Returns (x, y), each (n,)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.5, 1.5, n)
    y = np.sin(2.0 * x) + rng.normal(0.0, noise_std, n)
    return x, y


def polyfeatures(x, degree=3):
    """(N, degree+1) Vandermonde features (1, x, x^2, ..., x^degree)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** k for k in range(degree + 1)])


def plot_fit(theta, x, y, path):
    """Data scatter + the degree-3 fit theta, saved to path."""
    grid = np.linspace(-1.6, 1.6, 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, y, "o", ms=4, color="tab:gray", label="data")
    ax.plot(grid, polyfeatures(grid) @ theta, color="tab:red",
            label="degree-3 fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curves(curves, path, xlabel="iteration"):
    """Loss curves on one axes, log-y. curves: dict label -> (K,) loss
    array (labels may be learning rates or run names)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for label, losses in curves.items():
        losses = np.asarray(losses, dtype=float)
        ax.plot(np.arange(len(losses)), np.clip(losses, 1e-12, None),
                label=f"{label}")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE loss")
    ax.legend(title=r"$\alpha$" if xlabel == "iteration" else None,
              fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
# STUDENT SECTION: implement everything below.
# ======================================================================


def predict(theta, X):
    """Model predictions X @ theta. X: (N, k) features, theta: (k,).
    Returns (N,)."""
    raise NotImplementedError("implement predict() in gd.py")


def mse_loss(theta, X, y):
    """Mean squared error (1/N) sum (predict - y)^2, a float."""
    raise NotImplementedError("implement mse_loss() in gd.py")


def mse_grad(theta, X, y):
    """Analytic MSE gradient wrt theta: (2/N) X^T (X theta - y).
    Returns (k,). Derive it on paper first; no finite differences, no
    autograd (the autograder checks yours AGAINST finite differences)."""
    raise NotImplementedError("implement mse_grad() in gd.py")


def gd(theta0, grad_fn, loss_fn, alpha, iters):
    """Plain gradient descent: theta <- theta - alpha * grad_fn(theta),
    run iters times from theta0.

    grad_fn(theta) -> (k,) gradient; loss_fn(theta) -> float.
    Returns (theta_final, losses) where losses[k] is the loss BEFORE
    update k (so losses has length iters + 1 and ends with the final
    loss)."""
    raise NotImplementedError("implement gd() in gd.py")


# ======================================================================
# Provided driver (works once the student section above is done).
# ======================================================================


def main():
    x, y = make_curve_data()
    X = polyfeatures(x)
    curves = {}
    best_alpha, best_loss, best_theta = None, np.inf, None
    for alpha in config.LR_SWEEP:
        theta, losses = gd(np.zeros(X.shape[1]),
                           lambda th: mse_grad(th, X, y),
                           lambda th: mse_loss(th, X, y),
                           alpha, config.GD_ITERS)
        curves[alpha] = losses
        final = losses[-1]
        print(f"alpha={alpha}: final loss {final:.4g}")
        if np.isfinite(final) and final < best_loss:
            best_alpha, best_loss, best_theta = alpha, final, theta
    plot_loss_curves(curves, os.path.join(os.getcwd(),
                                          "ps2_loss_curves.png"))
    plot_fit(best_theta, x, y, os.path.join(os.getcwd(), "ps2_fit.png"))
    print(f"best alpha {best_alpha} (loss {best_loss:.4g}); wrote "
          "ps2_fit.png, ps2_loss_curves.png")


if __name__ == "__main__":
    main()
