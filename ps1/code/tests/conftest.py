"""PS1 autograder wiring (scripts/grader contract).

Registers the `public` marker (the released tier; the starter's
autograder.py wrapper runs exactly `-m public`) and makes the suite
runnable both through scripts/grader/run.py (which sets PYTHONPATH) and
bare (`GRADER_SUBMISSION=... pytest ps1/code/tests`), by
appending the repo paths the tests need:

  scripts/grader  gradelib, the only sanctioned way to import student code
  sim             coursesim
  starter         provided modules (grid_mdp, dubins, constants) as a
                  LAST-resort fallback, so a bare four-file submission
                  still imports; a submission's own copy wins because
                  gradelib puts the submission dir first during module
                  exec
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# In a student release, tests/ sits inside code/. Support the handout's
# direct per-task pytest commands without asking students to set grader
# environment variables. Staff submissions remain explicitly selected.
_CODE = os.path.dirname(_HERE)
if os.path.basename(_CODE) == "code":
    os.environ.setdefault("GRADER_SUBMISSION", _CODE)

for _p in (os.path.join(_ROOT, "scripts", "grader"),
           os.path.join(_ROOT, "sim"),
           os.path.join(_ROOT, "ps1", "code")):
    if _p not in sys.path:
        sys.path.append(_p)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "public: released tier; the starter autograder.py wrapper runs "
        "exactly this subset (-m public)")
