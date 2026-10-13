"""PS3 provided models: motion model and likelihood-field sensor model.

PROVIDED COMPLETE, DO NOT EDIT. The autograder always scores against the
course copy of this file, so local edits will not follow your submission.
Your filter implementation goes in particle_filter.py.

Both models are pure numpy and vectorized over the whole particle set:
one MotionModel.sample call pushes every particle, one
SensorModel.log_likelihood call scores every particle. Your filter should
make one vectorized call to each per step, rather than looping over
particles in Python.

Frames and units (SI, matching the simulator):
  particle state  (N, 3) [x, y, theta] world frame (m, m, rad)
  odom            (3,) [dx, dy, dtheta] relative motion in the PREVIOUS
                  pose's frame (the Obs.odom field)
  scan            (n_beams,) ranges in meters, Lidar.beam_angles order
"""
import numpy as np

from coursesim.dynamics import wrap_angle


class MotionModel:
    """Odometry motion model: apply the measured relative motion in each
    particle's own frame, plus additive Gaussian noise.

    The noise scales default LARGER than the simulator's odometry noise
    (xy 0.01 m, theta 0.005 rad per step): a particle filter needs its
    process noise to overbound the real noise, or the cloud collapses
    tighter than the odometry deserves and the true pose walks out of it.

    xy_std     meters, per-step noise on dx and dy (robot frame)
    theta_std  radians, per-step noise on dtheta
    """

    def __init__(self, xy_std=0.02, theta_std=0.01):
        self.xy_std = float(xy_std)
        self.theta_std = float(theta_std)

    def sample(self, states, odom, rng):
        """Push all particles through one noisy motion step.

        states  (N, 3) particle poses; NOT mutated
        odom    (3,) [dx, dy, dtheta], robot frame (Obs.odom)
        rng     np.random.Generator; consumes exactly ONE
                normal(size=(N, 3)) draw (the draw-order contract the
                seeded autograder tests rely on)

        Returns a NEW (N, 3) array. Per particle: perturb the odom with
        noise (in the robot frame), rotate [dx, dy] by the particle's own
        heading into the world frame, add, and wrap theta.
        """
        states = np.asarray(states, dtype=float)
        odom = np.asarray(odom, dtype=float)
        scales = np.array([self.xy_std, self.xy_std, self.theta_std])
        d = odom[None, :] + rng.normal(0.0, 1.0, states.shape) * scales
        c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
        out = np.empty_like(states)
        out[:, 0] = states[:, 0] + c * d[:, 0] - s * d[:, 1]
        out[:, 1] = states[:, 1] + s * d[:, 0] + c * d[:, 1]
        out[:, 2] = wrap_angle(states[:, 2] + d[:, 2])
        return out


class SensorModel:
    """Likelihood-field lidar model over the map's distance transform.

    For each particle and each beam that HIT something (range strictly
    under max_range), compute the world point where the beam should have
    ended if the particle were the true pose, look up
    map.distance_field at that point (meters to the nearest obstacle,
    precomputed in the Map class), and score it through a zero-mean
    Gaussian: endpoints on an obstacle score high, endpoints in open
    space score low. Max-range beams carry no endpoint and are skipped,
    the standard likelihood-field convention.

    The Gaussian is unnormalized (the constant cancels when the filter
    normalizes weights):

        log L(particle) = sum_hits  -0.5 * (d / sigma)^2

    map        coursesim.Map (its distance_field is the likelihood field)
    beam_angles (n_beams,) bearings relative to heading (Lidar.beam_angles)
    max_range  meters; beams reporting >= max_range are treated as misses
    sigma      meters, likelihood-field width. Wider forgives more pose
               error per beam but flattens the field the correct step
               climbs. The course default matches the shipped figure.
    """

    #: endpoints that fall off the map score as this many sigmas out
    _OOB_SIGMAS = 5.0

    def __init__(self, map, beam_angles, max_range, sigma=0.15):
        self.map = map
        self.beam_angles = np.asarray(beam_angles, dtype=float)
        self.max_range = float(max_range)
        self.sigma = float(sigma)

    def log_likelihood(self, states, scan):
        """Log-likelihood of one scan for every particle.

        states  (N, 3) particle poses
        scan    (n_beams,) measured ranges in beam_angles order

        Returns (N,) unnormalized log-likelihoods. Fully vectorized:
        one (N, n_hits) endpoint grid lookup, no Python loops.
        """
        states = np.asarray(states, dtype=float)
        scan = np.asarray(scan, dtype=float)
        hit = scan < self.max_range - 1e-6
        if not hit.any():
            return np.zeros(states.shape[0])
        r = scan[hit]                                     # (B,)
        angles = states[:, 2:3] + self.beam_angles[hit][None, :]  # (N, B)
        ex = states[:, 0:1] + r[None, :] * np.cos(angles)
        ey = states[:, 1:2] + r[None, :] * np.sin(angles)
        row, col = self.map.world_to_grid(ex, ey)
        h, w = self.map.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        rr = np.clip(row, 0, h - 1)
        cc = np.clip(col, 0, w - 1)
        d = np.where(inside, self.map.distance_field[rr, cc],
                     self._OOB_SIGMAS * self.sigma)
        return -0.5 * np.sum((d / self.sigma) ** 2, axis=1)
