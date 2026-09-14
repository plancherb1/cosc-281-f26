"""Pure kinematic-bicycle dynamics.

This function is shared by the simulator, planners, and controller prediction
models. Purity is the contract: no object state, globals, or RNG. The same
inputs produce the same output bit-for-bit.

State and units (F1 freeze surface):
  state = np.ndarray (3,) [x, y, theta]
    x, y   meters, world frame
    theta  radians, heading measured CCW from the world +x axis,
           wrapped to [-pi, pi)
  control = (v, steer)
    v      m/s, commanded speed along the heading (negative = reverse)
    steer  radians, front-wheel steering angle, positive = left turn
  dt in seconds.

The step clamps v to [-max_speed, max_speed] and steer to
[-max_steer, max_steer], then integrates the bicycle kinematics
    x' = v cos(theta),  y' = v sin(theta),  theta' = (v / wheelbase) tan(steer)
EXACTLY over dt (constant-control arc), so a held steer traces a circle of
radius wheelbase / tan(steer). New physics goes in new CarParams fields
with defaults; this signature is frozen at F1.
"""
import math
from dataclasses import dataclass

import numpy as np

# Below this curvature (1/m) the arc is integrated as a straight line to
# avoid dividing by a near-zero curvature.
_CURVATURE_EPS = 1e-6


@dataclass
class CarParams:
    """Car geometry and actuation limits. SI units.

    wheelbase  meters, front-to-rear axle distance
    max_steer  radians, symmetric steering limit; implies the minimum turn
               radius wheelbase / tan(max_steer)
    max_speed  m/s, symmetric speed limit (reverse allowed at -max_speed)
    radius     meters, footprint radius used for collision checks and for
               obstacle inflation in Map.occupancy_grid
    """
    wheelbase: float = 0.33
    max_steer: float = 0.42
    max_speed: float = 2.0
    radius: float = 0.15

    @property
    def min_turn_radius(self):
        """Minimum turning radius in meters, wheelbase / tan(max_steer)."""
        return self.wheelbase / math.tan(self.max_steer)


def wrap_angle(theta):
    """Wrap an angle (radians) to [-pi, pi). Works on scalars and arrays."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def step(state, control, params, dt):
    """Advance the kinematic bicycle by one step of dt seconds.

    state:   array-like (3,) [x, y, theta] (meters, meters, radians)
    control: (v, steer) in (m/s, radians); clamped to params limits
    params:  CarParams
    dt:      seconds

    Returns a NEW np.ndarray (3,); the input state is never mutated.
    Pure function: no RNG, no globals, no side effects.
    """
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    v = float(np.clip(control[0], -params.max_speed, params.max_speed))
    steer = float(np.clip(control[1], -params.max_steer, params.max_steer))

    kappa = math.tan(steer) / params.wheelbase  # path curvature, 1/m
    if abs(kappa) < _CURVATURE_EPS:
        x_new = x + v * dt * math.cos(theta)
        y_new = y + v * dt * math.sin(theta)
        theta_new = theta
    else:
        # Exact constant-curvature arc: the pose moves along a circle of
        # radius 1/kappa centered one radius to the side of the heading.
        theta_new = theta + v * kappa * dt
        x_new = x + (math.sin(theta_new) - math.sin(theta)) / kappa
        y_new = y - (math.cos(theta_new) - math.cos(theta)) / kappa
    return np.array([x_new, y_new, wrap_angle(theta_new)])
