"""Shared helpers for the staff autograder suites (scripts/grader).

Contract: run.py sets GRADER_SUBMISSION to the directory being graded (a
complete submission: every student-owned file, solved or student-written).
Test suites import student modules ONLY through load(), so one suite grades
the reference solution, buggy variants, and real submissions unchanged.
"""
import importlib.util
import os
import sys


def submission_dir():
    d = os.environ.get("GRADER_SUBMISSION")
    if not d:
        raise RuntimeError("GRADER_SUBMISSION is not set; run via run.py")
    return os.path.abspath(d)


def load(name):
    """Import <submission>/<name>.py as a fresh module.

    Fresh each call: suites may mutate module state (e.g. seeding), and a
    stale cached module from a previous variant run must never leak in.
    """
    path = os.path.join(submission_dir(), name + ".py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"submission is missing {name}.py ({path})")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    added = submission_dir() not in sys.path
    if added:
        sys.path.insert(0, submission_dir())
    try:
        spec.loader.exec_module(mod)
    finally:
        if added:
            sys.path.remove(submission_dir())
    return mod
