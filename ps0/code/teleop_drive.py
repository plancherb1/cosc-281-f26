"""PS0 C1: drive the car around and save the ride. PROVIDED, DO NOT EDIT.

Usage (from the repo root, inside the course venv):

    python ps0/code/teleop_drive.py --map sim/maps/open.yaml

Arrow keys (or wasd) drive, space stops, q quits. On quit the ground
truth trajectory is saved over the map as ps0_teleop.png (or --out).
Drive for at least 20 seconds of sim time; the window title shows the
clock. Speed is in m/s, steering in radians, both clamped to the course
car's limits.

The graded run requires an interactive Matplotlib window. ``--headless`` is
available only as an explicit installation diagnostic; its scripted trajectory
does not satisfy C1.
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
try:
    import coursesim  # noqa: F401  (installed or already on the path)
except ImportError:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "sim"))

from coursesim import Simulator

_HEADLESS_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo"}

# Scripted fallback: (key, hold_steps) pairs, 250 steps = 25 s at dt 0.1.
_SCRIPT = [("up", 30), ("up", 30), ("left", 40), ("right", 40),
           ("right", 40), ("left", 40), ("up", 30)]


def _interactive_backend_or_exit(matplotlib):
    """Select TkAgg before pyplot imports, or explain how to enable a GUI."""
    backend = matplotlib.get_backend().lower()
    if backend not in _HEADLESS_BACKENDS:
        return
    if (sys.platform.startswith("linux") and
            not os.environ.get("DISPLAY") and
            not os.environ.get("WAYLAND_DISPLAY")):
        sys.exit(
            "Cannot open the interactive driving window: no graphical "
            "display is available.\n\n"
            "If you are working locally, verify Tk with 'python3 -m tkinter' "
            "and install it on Ubuntu/Debian with:\n"
            "  sudo apt install python3-tk\n"
            "If you are connected over ssh, use ssh -X or run the exercise "
            "on your local computer.\n"
            "The --headless option is only a diagnostic and does not "
            "complete C1."
        )
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg", force=True)
        return
    except (ImportError, ModuleNotFoundError):
        pass
    sys.exit(
        "Cannot open the interactive driving window: Python's Tk GUI support "
        "is missing and Matplotlib selected the noninteractive "
        f"'{backend}' backend.\n\n"
        "Fix it, then run the same command again:\n"
        "  Ubuntu/Debian: sudo apt install python3-tk\n"
        "  macOS: install Python 3.12 from python.org (includes Tk)\n"
        "  Windows: modify the Python installation and enable Tcl/Tk\n"
        "If you deliberately set MPLBACKEND=Agg, unset it first.\n"
        "The --headless option is only a diagnostic and does not complete C1."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", required=True, help="path to a map .yaml")
    ap.add_argument("--out", default="ps0_teleop.png",
                    help="output image path (default ps0_teleop.png)")
    ap.add_argument("--seed", type=int, default=0,
                    help="simulator seed (default 0)")
    ap.add_argument("--headless", action="store_true",
                    help="run a scripted diagnostic without a window; "
                         "does not complete C1")
    args = ap.parse_args()

    import matplotlib
    if args.headless:
        matplotlib.use("Agg", force=True)
    else:
        _interactive_backend_or_exit(matplotlib)
    from coursesim.viz import MapView, Teleop

    sim = Simulator(args.map, seed=args.seed)
    teleop = Teleop(sim)

    output_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(output_parent, exist_ok=True)

    if args.headless:
        print("HEADLESS DIAGNOSTIC: running a scripted 25 s drive. "
              "This does not complete C1.")
        for key, hold in _SCRIPT:
            teleop.step_key(key)
            for _ in range(hold - 1):
                teleop.step_key(None)
        view = MapView(sim.map)
        view.add_trace(sim.ground_truth_trace)
        view.save(args.out)
    else:
        teleop.run(save_on_quit=args.out)

    trace = sim.ground_truth_trace
    print(f"saved {args.out}: {len(trace) - 1} steps, "
          f"{(len(trace) - 1) * sim.dt:.1f} s of sim time")


if __name__ == "__main__":
    main()
