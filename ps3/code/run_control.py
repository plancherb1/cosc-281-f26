"""Provided PS3 controller experiment and plot. Edit control_gains.py."""
import argparse
from pathlib import Path
import numpy as np

import control_gains
from control_experiment import evaluate, acceptance, metrics, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Figure directory (default: ps3/figures).")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = args.out or here.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = evaluate(control_gains)
    checks = acceptance(results)
    for name, result in results.items():
        print(name, metrics(result))
        if result["gain"] is not None:
            print("  computed LQR gain:", result["gain"])
    for name, passed in checks.items():
        print(("PASS " if passed else "ADJUST ") + name)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for name in ("pid_overshoot", "pid_slow"):
        result = results[name]
        axes[0, 0].plot(result["time"], result["states"][:, 0], label=name)
    for name in ("lqr_responsive", "lqr_gentle"):
        result = results[name]
        axes[0, 1].plot(result["time"], result["states"][:, 0], label=name)
    loaded = {
        "PID with integral": results["pid_reject"],
        "same PID, integral disabled": simulate("pid", control_gains.PID_REJECT,
                                              disturbance=True, integral=False),
        "responsive LQR": simulate("lqr", control_gains.LQR_RESPONSIVE,
                                   disturbance=True),
    }
    for name, result in loaded.items():
        axes[1, 0].plot(result["time"], result["states"][:, 0], label=name)
        axes[1, 1].plot(result["time"][:-1], result["controls"], label=name)
    for ax, title in zip(axes.flat, ("PID responses", "LQR responses",
                                   "Constant disturbance from 8 s",
                                   "Commanded acceleration under disturbance")):
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Acceleration (m/s²)")
    axes[1, 0].axvline(8.0, color="gray", linestyle=":")
    axes[1, 1].axvline(8.0, color="gray", linestyle=":")
    path = out / "ps3_control.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}. Review the curves even when all checks pass.")


if __name__ == "__main__":
    main()
