"""Released public tests; staff-only checks are omitted."""
import numpy as np

import pytest

import gradelib

import staff_common

CASES = ("worked", "padded", "rand_p0s1", "rand_p1s2", "rand_p2s1", "box")

def _case(name):
    refs = staff_common.conv_refs()
    return (refs[f"{name}_image"], refs[f"{name}_kernel"],
            int(refs[f"{name}_pad"]), int(refs[f"{name}_stride"]),
            refs[f"{name}_expected"])

@pytest.mark.public
def test_worked_weighted_sum():
    conv = gradelib.load("conv")
    image, kernel, pad, stride, expected = _case("worked")
    out = np.asarray(conv.convolve2d(image, kernel, pad, stride))
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, atol=1e-8)

@pytest.mark.public
def test_displayed_filter_coefficients():
    conv = gradelib.load("conv")
    image = np.arange(1, 10, dtype=float).reshape(3, 3)
    kernel = np.zeros((3, 3))
    kernel[0, 0] = 2
    kernel[2, 2] = -1
    out = np.asarray(conv.convolve2d(image, kernel))
    assert out.shape == (1, 1)
    np.testing.assert_allclose(out, [[2*1-9]], atol=1e-8)

@pytest.mark.public
@pytest.mark.parametrize("name", CASES)
def test_output_shapes(name):
    conv = gradelib.load("conv")
    refs = staff_common.conv_refs()
    image, kernel = refs[f"{name}_image"], refs[f"{name}_kernel"]
    pad, stride = int(refs[f"{name}_pad"]), int(refs[f"{name}_stride"])
    out = np.asarray(conv.convolve2d(image, kernel, pad, stride))
    assert out.shape == tuple(refs[f"{name}_shape"])

