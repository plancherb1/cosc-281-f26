"""Paths and public-test marker for ps2."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

for _p in (os.path.join(_ROOT, "scripts", "grader"),
           os.path.join(_ROOT, "sim"),
           os.path.join(_ROOT, "ps2", "code")):
    if _p not in sys.path:
        sys.path.append(_p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "public: released tier; the starter autograder.py wrapper runs "
        "exactly this subset (-m public)")
