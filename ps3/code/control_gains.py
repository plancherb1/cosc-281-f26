"""PS3 Task 2: select controller settings for the provided experiment.

Edit only the five tuples below. Run:
    python ps3/code/run_control.py

PID tuples are (kp, ki, kd). Useful exploration ranges are
kp: 0.2--4, ki: 0--1, kd: 0.2--4.
Start with ki=0 when studying overshoot and damping. Increasing kd often
reduces oscillation but can make the response slow. Integral action can
remove a persistent offset from the constant disturbance.

LQR tuples are (position weight, velocity weight, effort weight).
Try positive weights from 0.1--10. The provided solver computes K.
A larger velocity penalty often gives a slower, less oscillatory response.
Changing all weights by the same factor does not change the LQR policy.

These are exploration ranges, not grading bounds. Many settings pass.
The initial values below run, but do not meet every requested response.
"""

# TODO: visible overshoot followed by convergence, with no disturbance.
PID_OVERSHOOT = (1.0, 0.0, 1.0)

# TODO: deliberately slow, nearly non-overshooting response, with no disturbance.
PID_SLOW = (1.0, 0.0, 1.0)

# TODO: recover to the reference after the constant load begins at 8 seconds.
PID_REJECT = (1.0, 0.0, 1.0)

# TODO: prompt LQR response with modest overshoot, with no disturbance.
LQR_RESPONSIVE = (1.0, 1.0, 1.0)

# TODO: slower LQR response with very little overshoot, with no disturbance.
LQR_GENTLE = (1.0, 1.0, 1.0)
