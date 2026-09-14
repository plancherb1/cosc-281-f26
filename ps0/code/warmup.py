"""PS0, Part 2, C2: five mathematical warmup functions.

This is the only code file you edit and submit. Each function implements a
specific calculation from Part 1 of the handout. Read its contract, translate
the displayed formula into NumPy, and run the public tests from the repository
root with

    python ps0/code/autograder.py
"""
import numpy as np


def matvec(A, v):
    """Compute the matrix-vector product from written problem P1(a).

    Args:
        A: An (m, n) array-like matrix.
        v: An (n,) array-like vector.

    Returns:
        An (m,) float array ``out`` satisfying

            out[i] = sum(A[i, j] * v[j] for j in range(n)).

    For the P1(a) values, ``matvec([[2, -1], [0, 3]], [2, 1])`` returns
    ``array([3., 3.])``.

    Use two explicit nested loops. NumPy may be used to convert the inputs,
    inspect their shapes, and allocate the output, but multiplication and
    accumulation must be scalar operations inside the loops. Do not use
    ``@``, ``np.dot``, ``np.matmul``, ``np.einsum``, or another routine that
    computes the product for you. Convert both inputs with
    ``np.asarray(..., dtype=float)`` before indexing them.
    """
    raise NotImplementedError("implement matvec() in warmup.py")


def rot2d(theta):
    """Construct the counterclockwise rotation matrix from P1(b).

    Args:
        theta: An angle in radians.

    Returns:
        The (2, 2) float array

            [[cos(theta), -sin(theta)],
             [sin(theta),  cos(theta)]].

    Thus ``rot2d(np.pi / 2) @ [1, 0]`` is approximately ``[0, 1]``.
    The minus sign belongs on the sine term in the first row.
    """
    raise NotImplementedError("implement rot2d() in warmup.py")


def bayes_posterior(prior, p_pos_given_true, p_pos_given_false):
    """Generalize the one-reading Bayes update from P2(b).

    Args:
        prior: P(event is true) before the reading.
        p_pos_given_true: P(positive reading | event is true).
        p_pos_given_false: P(positive reading | event is false).

    Returns:
        The posterior probability

            p_pos_given_true * prior
            ----------------------------------------------- .
            p_pos_given_true * prior
              + p_pos_given_false * (1 - prior)

    All three inputs lie in [0, 1], and the denominator is positive. For the
    P2(b) values, ``bayes_posterior(1/3, 0.9, 0.3)`` returns ``0.6``.
    """
    raise NotImplementedError("implement bayes_posterior() in warmup.py")


def expectation(values, probs):
    """Generalize the discrete expectation from P2(a).

    Args:
        values: A (k,) array-like containing the possible outcomes.
        probs: A (k,) array-like containing their probabilities; entries sum
            to one.

    Returns:
        The float ``sum(values[i] * probs[i] for i in range(k))``.

    For the P2(a) values, ``expectation([0, 2, 4],
    [0.5, 0.25, 0.25])`` returns ``1.5``. Elementwise multiplication followed
    by a sum is sufficient; return a plain float.
    """
    raise NotImplementedError("implement expectation() in warmup.py")


def central_diff_grad(f, x, eps=1e-6):
    """Implement the coordinate-wise numerical check from P3(a).

    Args:
        f: A callable that maps one (n,) array to a scalar.
        x: The (n,) array-like point where the gradient is estimated.
        eps: A small positive displacement.

    Returns:
        An (n,) float array ``grad`` satisfying

            grad[i] = (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps),

        where ``e_i`` is the i-th standard basis vector.

    For ``f(x) = x[0]**2 + 3*x[0]*x[1]`` and ``x = [1, 2]``, the result is
    approximately ``[8, 3]``, matching P3(a). Convert ``x`` to floating point,
    make fresh perturbed copies for each coordinate, and divide by ``2*eps``.
    """
    raise NotImplementedError("implement central_diff_grad() in warmup.py")
