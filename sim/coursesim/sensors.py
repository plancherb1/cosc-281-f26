"""Lidar and odometry sensor models.

Both sensors take a numpy Generator at construction and draw ALL of their
randomness from it (one child stream each, spawned by the Simulator per
rng.py). Noise parameters are constructor arguments so experiments can
vary them without changing the implementation.

Draw-order contract (frozen; rng.py documents the per-step schedule):
  Lidar.scan     always draws normal(size=n_beams) then uniform(size=n_beams),
                 even when noise_std or dropout_p is 0, so turning a noise
                 knob never shifts the stream.
  Odometry.measure always draws normal(size=3).
"""
import numpy as np

from .dynamics import wrap_angle


class Lidar:
    """2-D range scanner: batched fixed-step ray marching over the
    occupancy grid.

    n_beams beams at fixed bearings evenly spaced over [-pi, pi) relative
    to the car heading (beam_angles[0] = -pi points backward; the beam at
    bearing 0 points along the heading). All beams march together as one
    numpy batch in steps of half a grid cell, so range error is about half
    a resolution plus the cell-center geometry of the grid.

    Constructor arguments:
      n_beams    number of beams
      max_range  meters; beams that hit nothing return exactly max_range
      noise_std  meters, std of additive Gaussian range noise on hits
      dropout_p  probability a beam drops (returns max_range)
      rng        np.random.Generator, the sensor's ONLY randomness source
    """

    def __init__(self, n_beams=36, max_range=5.0, noise_std=0.02,
                 dropout_p=0.0, rng=None):
        if rng is None:
            raise ValueError("Lidar requires an explicit rng Generator")
        self.n_beams = int(n_beams)
        self.max_range = float(max_range)
        self.noise_std = float(noise_std)
        self.dropout_p = float(dropout_p)
        self.rng = rng
        # Fixed beam bearings relative to the car heading, radians.
        self.beam_angles = np.linspace(-np.pi, np.pi, self.n_beams,
                                       endpoint=False)

    def scan(self, map, pose):
        """Scan the map from a pose. pose: (3,) [x, y, theta] (m, m, rad).

        Returns np.ndarray (n_beams,) of ranges in meters, in beam_angles
        order. A beam reports the distance to the first occupied cell along
        its ray; max_range exactly if nothing was hit within range or the
        beam dropped out. Noisy ranges are clipped to [0, max_range].
        Consumes exactly one normal(n_beams) and one uniform(n_beams) draw.
        """
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        angles = theta + self.beam_angles                      # (B,)
        step = 0.5 * map.resolution
        ts = np.arange(1, int(self.max_range / step) + 1) * step  # (S,)
        # Sample points for every beam at every march distance: (S, B).
        xs = x + ts[:, None] * np.cos(angles)[None, :]
        ys = y + ts[:, None] * np.sin(angles)[None, :]
        blocked = ~map.is_free(xs, ys)                         # (S, B)
        hit_any = blocked.any(axis=0)                          # (B,)
        first = np.argmax(blocked, axis=0)                     # (B,)
        ranges = np.where(hit_any, ts[first], self.max_range)

        noise = self.rng.normal(0.0, 1.0, self.n_beams) * self.noise_std
        drop = self.rng.uniform(size=self.n_beams) < self.dropout_p
        ranges = np.where(hit_any, ranges + noise, ranges)
        ranges = np.where(drop, self.max_range, ranges)
        return np.clip(ranges, 0.0, self.max_range)


class OdometryNoise:
    """Additive Gaussian noise scales for one odometry measurement.
    xy_std in meters, theta_std in radians, both per step."""

    def __init__(self, xy_std=0.01, theta_std=0.005):
        self.xy_std = float(xy_std)
        self.theta_std = float(theta_std)


class Odometry:
    """Noisy relative-motion sensor.

    measure(prev_pose, pose) returns the true motion between two poses
    expressed in the PREVIOUS pose's frame, plus additive Gaussian noise:
    np.ndarray (3,) [dx, dy, dtheta] in (m, m, rad), dtheta wrapped to
    [-pi, pi). This is the Obs.odom field and is expressed in the frame
    expected by downstream motion models.

    noise_params: OdometryNoise. rng: np.random.Generator, the sensor's
    ONLY randomness source (one normal(3) draw per measure).
    """

    def __init__(self, noise_params=None, rng=None):
        if rng is None:
            raise ValueError("Odometry requires an explicit rng Generator")
        self.noise_params = noise_params or OdometryNoise()
        self.rng = rng

    def measure(self, prev_pose, pose):
        """Relative motion from prev_pose to pose in prev_pose's frame,
        with noise. Both poses: (3,) [x, y, theta]."""
        dx_w = float(pose[0]) - float(prev_pose[0])
        dy_w = float(pose[1]) - float(prev_pose[1])
        c, s = np.cos(prev_pose[2]), np.sin(prev_pose[2])
        # Rotate the world-frame displacement by -theta_prev.
        dx = c * dx_w + s * dy_w
        dy = -s * dx_w + c * dy_w
        dtheta = wrap_angle(float(pose[2]) - float(prev_pose[2]))
        scales = np.array([self.noise_params.xy_std, self.noise_params.xy_std,
                           self.noise_params.theta_std])
        noise = self.rng.normal(0.0, 1.0, 3) * scales
        out = np.array([dx, dy, dtheta]) + noise
        out[2] = wrap_angle(out[2])
        return out
