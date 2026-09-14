"""PS0 C2 gate: the five warmup functions, checked against the drill
numbers they mirror plus randomized cross-checks vs numpy. ALL tests are
public: a review pset has nothing to hide, so the starter wrapper shows
students exactly what submission runs. Seeded per the harness
determinism rule; budget is trivially inside 5 minutes.
"""
import ast
import os

import numpy as np
import pytest

import gradelib

pytestmark = pytest.mark.public


@pytest.fixture()
def warmup():
    return gradelib.load("warmup")


def test_matvec_drill_numbers(warmup):
    # written P1(a): A v = (3, 3)
    out = warmup.matvec([[2, -1], [0, 3]], [2, 1])
    assert np.allclose(out, [3, 3])


def test_matvec_matches_numpy_including_nonsquare(warmup):
    source_path = os.path.join(gradelib.submission_dir(), "warmup.py")
    with open(source_path, encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=source_path)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and
                    node.name == "matvec")
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    has_nested_loops = any(
        any(isinstance(child, ast.For) and child is not outer
            for child in ast.walk(outer))
        for outer in loops)
    assert has_nested_loops, "matvec must use two explicit nested for loops"

    forbidden_names = {"dot", "matmul", "einsum", "tensordot", "inner",
                       "vdot", "multi_dot"}
    assert not any(isinstance(node, ast.BinOp) and
                   isinstance(node.op, ast.MatMult)
                   for node in ast.walk(function)), \
        "matvec may not use the @ operator"
    assert not any(
        isinstance(node, ast.Call) and
        ((isinstance(node.func, ast.Attribute) and
          node.func.attr in forbidden_names) or
         (isinstance(node.func, ast.Name) and
          node.func.id in forbidden_names))
        for node in ast.walk(function)), \
        "matvec must perform scalar multiplication and accumulation itself"

    rng = np.random.default_rng(0)
    for shape in ((3, 3), (2, 3), (4, 2)):
        A = rng.standard_normal(shape)
        v = rng.standard_normal(shape[1])
        assert np.allclose(warmup.matvec(A, v), A @ v)


def test_rot2d_written_quarter_turn_and_orthogonality(warmup):
    # written P1(b): B = R(pi/2)
    R = warmup.rot2d(np.pi / 2)
    assert np.allclose(R @ np.array([1.0, 0.0]), [0.0, 1.0], atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rot2d_composition(warmup):
    a, b = 0.4, 1.1
    assert np.allclose(warmup.rot2d(a) @ warmup.rot2d(b),
                       warmup.rot2d(a + b), atol=1e-12)


def test_bayes_drill_numbers(warmup):
    # written P2(b): beep -> posterior 0.6
    assert warmup.bayes_posterior(1 / 3, 0.9, 0.3) == pytest.approx(0.6)


def test_bayes_uninformative_sensor_keeps_prior(warmup):
    assert warmup.bayes_posterior(0.25, 0.7, 0.7) == pytest.approx(0.25)


def test_expectation_drill_numbers(warmup):
    # written P2(a): E[X] = 1.5
    assert warmup.expectation([0, 2, 4],
                              [0.5, 0.25, 0.25]) == pytest.approx(1.5)


def test_expectation_linearity(warmup):
    vals = np.array([0.0, 2.0, 4.0])
    probs = np.array([0.5, 0.25, 0.25])
    e = warmup.expectation(vals, probs)
    assert warmup.expectation(2 * vals + 1, probs) == pytest.approx(2 * e + 1)


def test_gradient_quadratic(warmup):
    g = warmup.central_diff_grad(lambda x: float(np.sum(x ** 2)),
                                 np.array([1.0, -2.0]))
    assert np.allclose(g, [2.0, -4.0], atol=1e-3)


def test_gradient_written_central_difference(warmup):
    # written P3(a), including its deliberately coarse eps = 0.1 check
    f = lambda x: float(x[0] ** 2 + 3 * x[0] * x[1])
    g = warmup.central_diff_grad(f, [1, 2], eps=0.1)
    assert np.allclose(g, [8.0, 3.0], atol=1e-12)
