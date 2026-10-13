"""PS3 Task 1: particle-filter localization. YOUR CODE GOES HERE.

You implement three methods: predict, correct, resample. Everything else
(initialization, the estimate, the models) is provided. The runner's loop
reads as the Bayes filter cycle verbatim:

    obs = sim.step(control)
    pf.predict(obs.odom)      # push belief through the motion model
    pf.correct(obs.scan)      # reweight by measurement likelihood
    est = pf.estimate()
    pf.resample()             # reallocate the particle budget

Implementation requirements:
  - ALL randomness comes from self.rng. Touching np.random directly
    scores zero on the seeded tests.
  - sim.ground_truth_pose is for plotting and grading only. An estimator
    that reads it scores zero.
  - predict/correct/resample mutate the filter in place and return None.
"""
import numpy as np

from coursesim.dynamics import wrap_angle


class ParticleFilter:
    """Particle filter over (x, y, theta) with N weighted particles.

    map           coursesim.Map (the known world for localization)
    n_particles   number of particles N
    rng           np.random.Generator, the filter's SOLE randomness
                  source (spawned off the simulator seed by the runner)
    motion_model  models.MotionModel (provided)
    sensor_model  models.SensorModel (provided)
    init_pose     optional (3,) [x, y, theta]. None initializes the cloud
                  uniformly over free space (global localization); a pose
                  initializes a Gaussian cloud around it (tracking, what
                  the PS3 runners use: the car knows its starting box).
    init_std      (xy_std_m, theta_std_rad) for the Gaussian init.

    State (the three arrays your methods read and write):
      self.particles  (N, 3) float [x, y, theta] world frame (m, m, rad)
      self.weights    (N,) float, nonnegative, sums to 1
      self.rng        the Generator
    """

    def __init__(self, map, n_particles, rng, motion_model, sensor_model,
                 init_pose=None, init_std=(0.3, 0.1)):
        # PROVIDED, DO NOT EDIT. Draw-order contract (seeded tests rely
        # on it): uniform init consumes integers(N) + uniform((N, 2)) +
        # uniform(N); Gaussian init consumes ONE normal(size=(N, 3)).
        self.map = map
        self.n_particles = int(n_particles)
        self.rng = rng
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        n = self.n_particles
        if init_pose is None:
            free_rows, free_cols = np.nonzero(~map.occupancy)
            pick = rng.integers(0, free_rows.size, n)
            x, y = map.grid_to_world(free_rows[pick], free_cols[pick])
            jitter = rng.uniform(-0.5, 0.5, (n, 2)) * map.resolution
            theta = rng.uniform(-np.pi, np.pi, n)
            self.particles = np.column_stack(
                [x + jitter[:, 0], y + jitter[:, 1], theta])
        else:
            init_pose = np.asarray(init_pose, dtype=float)
            xy_std, theta_std = init_std
            scales = np.array([xy_std, xy_std, theta_std])
            self.particles = init_pose[None, :] \
                + rng.normal(0.0, 1.0, (n, 3)) * scales
            self.particles[:, 2] = wrap_angle(self.particles[:, 2])
        self.weights = np.full(n, 1.0 / n)

    # ------------------------------------------------------ STUDENT CODE

    def predict(self, odom):
        """Push every particle through the motion model with noise.

        odom: (3,) relative motion [dx, dy, dtheta] in the robot frame
        (the Obs.odom field, meters and radians).

        Mutates self.particles; self.weights untouched. One call to the
        provided self.motion_model.sample (vectorized over all particles,
        drawing its noise from self.rng) does the work; your job is to
        wire it correctly.
        """
        # TODO(student): one motion_model.sample call.
        raise NotImplementedError

    def correct(self, scan):
        """Reweight particles by the measurement likelihood.

        scan: (n_beams,) lidar ranges (the Obs.scan field).

        Mutates self.weights: multiply each weight by its particle's
        likelihood, then renormalize so the weights sum to 1. The
        provided self.sensor_model.log_likelihood returns LOG-likelihoods
        (one vectorized call for all particles); subtract the max
        log-likelihood before exponentiating, to reduce the risk of numerical underflow. self.particles untouched.
        """
        # TODO(student): log_likelihood -> subtract max -> exp ->
        # multiply into weights -> renormalize.
        raise NotImplementedError

    def resample(self, u0=None):
        """Systematic resampling using cumulative weight intervals:

          cdf_j = w_0 + ... + w_j            (cumulative sum)
          u_i   = u0 + i / N                 for i = 0 .. N-1
          pick, for each u_i in order, the FIRST index j with
          cdf_j > u_i                        (STRICT inequality)

        u0: float in [0, 1/N), or None. When None, draw it as
        self.rng.uniform(0, 1/N). The autograder injects u0 to make its
        unit test bitwise; your filter loop just calls resample().

        Mutates self.particles (the selected copies, in u_i order) and
        resets self.weights to exactly 1/N. A cumulative weight is the right endpoint of a particle's interval.
        A point exactly on that boundary belongs to the interval on its
        right. For weights (0.2, 0.5, 0.3), cumulative weights are
        (0.2, 0.7, 1.0): points 0.1, 0.2, 0.8 select indices 0, 1, 2.
        Use np.searchsorted(cdf, points, side="right"); set cdf[-1] = 1.0
        to close any floating-point roundoff gap.
        """
        # TODO(student): cdf, the N draws, the strict-inequality
        # selection (np.searchsorted does it in one call; check side=),
        # then reset the weights.
        raise NotImplementedError

    # --------------------------------------------------------- PROVIDED

    def estimate(self):
        """(3,) weighted mean pose. PROVIDED, DO NOT EDIT.

        x and y are weighted means; theta is the circular weighted mean
        (atan2 of mean sin and cos), because averaging angles across the
        +-pi seam any other way invents poses that face backward.
        """
        w = self.weights
        x = float(w @ self.particles[:, 0])
        y = float(w @ self.particles[:, 1])
        theta = float(np.arctan2(w @ np.sin(self.particles[:, 2]),
                                 w @ np.cos(self.particles[:, 2])))
        return np.array([x, y, theta])
