"""Run the public PS2 tests against this code directory."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(PS_DIR)
SUITE = os.path.join(HERE, "tests")


def main():
    if not os.path.isdir(SUITE):
        sys.exit(f"cannot find the public tests at {SUITE}")
    env = dict(os.environ)
    env["GRADER_SUBMISSION"] = HERE
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (HERE, os.path.join(ROOT, "sim"), SUITE,
                    env.get("PYTHONPATH")) if p)
    env.setdefault("MPLBACKEND", "Agg")
    cmd = [sys.executable, "-m", "pytest", SUITE, "-q", "-m", "public",
           "-p", "no:cacheprovider"]
    sys.exit(subprocess.call(cmd, env=env, cwd=ROOT))


if __name__ == "__main__":
    main()
