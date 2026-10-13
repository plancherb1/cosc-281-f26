"""Provided PS3 response experiment. Students edit control_gains.py only.

The car is reduced to straight-line position and velocity, with commanded
acceleration. This supplied linear model is a teaching approximation, not
the full simulator's velocity-command bicycle model.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if HERE.name in ("starter", "solution") else HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))

import numpy as np
from coursesim.control import riccati_lqr

DT = 0.05
DURATION = 20.0
TARGET = 1.0
A = np.array([[1.0, DT], [0.0, 1.0]])
B = np.array([0.5 * DT**2, DT])
LIMIT = 3.0
CASES = {
    "pid_overshoot": ("pid", "PID_OVERSHOOT"),
    "pid_slow": ("pid", "PID_SLOW"),
    "pid_reject": ("pid", "PID_REJECT"),
    "lqr_responsive": ("lqr", "LQR_RESPONSIVE"),
    "lqr_gentle": ("lqr", "LQR_GENTLE"),
}


def parameters(values, kind):
    """Validate the three supplied gains or cost weights."""
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Each setting must contain three finite numbers.")
    if kind == "pid":
        if values[0] <= 0 or np.any(values[1:] < 0):
            raise ValueError("PID requires kp > 0 and ki, kd >= 0.")
    elif kind == "lqr":
        if np.any(values <= 0):
            raise ValueError("LQR requires positive position, velocity, and effort weights.")
    else:
        raise ValueError("Controller kind must be 'pid' or 'lqr'.")
    return values


def simulate(kind, values, *, disturbance=False, integral=True):
    """Run a 1 m reference change from rest and return states and controls.

    The disturbance is -0.2 m/s^2 from t=8 s onward. It is not supplied to
    the controller. The PID derivative term uses measured velocity, so a
    reference change does not create a derivative impulse. Conditional
    integration prevents accumulating error that drives actuator saturation.
    """
    values = parameters(values, kind)
    steps = round(DURATION / DT)
    states = np.zeros((steps + 1, 2))
    controls = np.zeros(steps)
    accumulated = 0.0
    gain = None
    if kind == "lqr":
        qp, qv, r = values
        Q = np.diag([qp, qv])
        # Reuse the first gain of a long-horizon solution at each time.
        # No Riccati implementation or gain derivation is required.
        gain = riccati_lqr(A, B[:, None], Q, [[r]], Q, 400)[0][0].ravel()
    for k in range(steps):
        position, velocity = states[k]
        error = TARGET - position
        if kind == "pid":
            kp, ki, kd = values
            if not integral:
                ki = 0.0
            candidate = accumulated + error * DT
            raw = kp * error + ki * candidate - kd * velocity
            if abs(raw) <= LIMIT or raw * error < 0:
                accumulated = candidate
            command = kp * error + ki * accumulated - kd * velocity
        else:
            command = -gain @ np.array([position - TARGET, velocity])
        controls[k] = np.clip(command, -LIMIT, LIMIT)
        load = -0.2 if disturbance and k * DT >= 8.0 else 0.0
        states[k + 1] = A @ states[k] + B * (controls[k] + load)
    if not np.all(np.isfinite(states)):
        raise ValueError("The response is not finite. Check the controller parameters.")
    return {
        "time": np.arange(steps + 1) * DT,
        "states": states,
        "controls": controls,
        "gain": gain,
    }


def metrics(result):
    """Report broad response properties, not a required reference trajectory."""
    position = result["states"][:, 0]
    reached = np.flatnonzero(position >= 0.8 * TARGET)
    return {
        "overshoot": float(max(0.0, position.max() - TARGET)),
        "tail_error": float(np.max(np.abs(position[-80:] - TARGET))),
        "rise80": float(result["time"][reached[0]]) if len(reached) else float("inf"),
        "peak_position": float(np.max(np.abs(position))),
        "peak_effort": float(np.max(np.abs(result["controls"]))),
    }


def evaluate(config):
    """Simulate the five submitted settings, including the disturbance case."""
    results = {}
    for name, (kind, field) in CASES.items():
        results[name] = simulate(kind, getattr(config, field),
                                 disturbance=(name == "pid_reject"))
    return results


def acceptance(results):
    """Public, deliberately broad completion checks for each requested shape."""
    m = {name: metrics(result) for name, result in results.items()}
    checks = {}
    for name, values in m.items():
        checks[name + "_bounded"] = values["peak_position"] <= 3.0
    checks["pid_visible_overshoot"] = 0.08 <= m["pid_overshoot"]["overshoot"] <= 0.70
    checks["pid_overshoot_settles"] = m["pid_overshoot"]["tail_error"] <= 0.10
    checks["pid_slow_shape"] = (m["pid_slow"]["overshoot"] <= 0.03
                              and 4.0 <= m["pid_slow"]["rise80"] <= 15.0
                              and m["pid_slow"]["tail_error"] <= 0.20)
    checks["pid_rejects_load"] = m["pid_reject"]["tail_error"] <= 0.05
    checks["lqr_responsive_shape"] = (m["lqr_responsive"]["rise80"] <= 4.0
                                    and m["lqr_responsive"]["overshoot"] <= 0.25
                                    and m["lqr_responsive"]["tail_error"] <= 0.10)
    checks["lqr_gentle_shape"] = (m["lqr_gentle"]["overshoot"] <= 0.03
                                and m["lqr_gentle"]["rise80"] >= m["lqr_responsive"]["rise80"] + 1.5
                                and m["lqr_gentle"]["rise80"] <= 15.0
                                and m["lqr_gentle"]["tail_error"] <= 0.20)
    return checks
