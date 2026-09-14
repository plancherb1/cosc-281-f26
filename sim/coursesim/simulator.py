"""Simulator: the student-facing fixed-step interface.

Synchronous, fixed-dt step loop. No wall clock, no rendering, no display:
a student can single-step it in a debugger and an autograder can replay it
bit-for-bit from the seed. Rendering lives in viz.py and is never a
dependency of stepping.

    sim = Simulator("sim/maps/maze.yaml", seed=478, car_params=CarParams())
    obs = sim.reset()
    obs = sim.step((v, steer))

Ground truth (`ground_truth_pose`, `ground_truth_trace`) is exposed for
PLOTTING AND AUTOGRADING ONLY. Estimators that read it score zero; the
handouts say so.
"""
from dataclasses import dataclass

import numpy as np

from . import dynamics, rng as rng_mod
from .camera import Camera
from .sensors import Lidar, Odometry, OdometryNoise
from .world import Map


@dataclass
class Obs:
    """One observation, returned by reset() and step().

    Field order and names form the stable course API.

      odom      np.ndarray (3,) [dx, dy, dtheta], noisy motion since the
                previous observation, in the previous pose's frame
                (m, m, rad). All zeros at reset.
      scan      np.ndarray (n_beams,) lidar ranges in meters, in
                Lidar.beam_angles order (bearings relative to heading).
      dt        seconds per step, fixed for the whole episode.
      t         seconds since reset (0.0 at reset).
      collided  True when this step's motion was blocked by an obstacle
                (the car did not move; see Simulator.step).
    """
    odom: np.ndarray
    scan: np.ndarray
    dt: float
    t: float
    collided: bool = False


class Simulator:
    """The course simulator: one car, one map, fixed-dt synchronous loop.

    Constructor arguments:
      map_path        path to a map .yaml (or a Map instance)
      seed            int; roots the episode's SeedSequence (rng.py)
      car_params      dynamics.CarParams (defaults = the course car)
      dt              seconds per step, fixed per episode (default 0.1)
      lidar_params    dict of Lidar constructor overrides, e.g.
                      {"n_beams": 72, "noise_std": 0.05} (no "rng" key;
                      the simulator wires the stream)
      odometry_noise  sensors.OdometryNoise override
      init_pose_std   (xy_std_m, theta_std_rad) Gaussian perturbation of
                      the start pose at reset. Default (0.0, 0.0): starts
                      exactly at map.start. The perturbation stream is
                      drawn either way, so dialing this never shifts the
                      sensor streams.
      camera_params   dict of Camera constructor overrides, e.g.
                      {"fov_deg": 75.0} (no "rng" key; the simulator
                      wires the reserved camera stream). Additive
                      post-F1 argument; the camera draws nothing unless
                      render_camera() is called.

    Attributes: map (the loaded Map), car_params, dt, lidar, odometry,
    camera. __init__ performs a reset(), so the attributes are live
    immediately; calling reset() again restarts the SAME episode
    bit-for-bit.
    """

    def __init__(self, map_path, seed=0, car_params=None, dt=0.1,
                 lidar_params=None, odometry_noise=None,
                 init_pose_std=(0.0, 0.0), camera_params=None):
        self.map = map_path if isinstance(map_path, Map) else Map.load(map_path)
        self.seed = seed
        self.car_params = car_params or dynamics.CarParams()
        self.dt = float(dt)
        self._lidar_params = dict(lidar_params or {})
        self._odometry_noise = odometry_noise or OdometryNoise()
        self._init_pose_std = tuple(init_pose_std)
        self._camera_params = dict(camera_params or {})
        self.reset()

    def reset(self):
        """Restart the episode from the seed. Deterministic: every stream
        is rebuilt from the same SeedSequence, so reset() replays the
        identical episode. Returns the initial Obs (zero odom, real scan,
        t = 0)."""
        streams, self._root_seq = rng_mod.make_streams(self.seed)
        self.lidar = Lidar(rng=streams["lidar"], **self._lidar_params)
        self.odometry = Odometry(self._odometry_noise, rng=streams["odometry"])
        # The camera owns the RESERVED `camera` stream (rng.py index 3).
        # Constructing it draws nothing; only render_camera() draws, so
        # episodes that never render replay bit-identically either way.
        self.camera = Camera(rng=streams["camera"], **self._camera_params)

        xy_std, theta_std = self._init_pose_std
        perturb = streams["init_pose"].normal(0.0, 1.0, 3) \
            * np.array([xy_std, xy_std, theta_std])
        pose = self.map.start + perturb
        pose[2] = dynamics.wrap_angle(pose[2])
        self._pose = pose
        self._trace = [pose.copy()]
        self._t = 0.0

        scan = self.lidar.scan(self.map, self._pose)
        return Obs(odom=np.zeros(3), scan=scan, dt=self.dt, t=self._t)

    def step(self, control):
        """Advance one step of dt seconds. control = (v, steer) in
        (m/s, rad); dynamics.step clamps both to the car limits.

        Collision rule: if the new pose's footprint (car_params.radius)
        would intersect an obstacle, the car does NOT move this step and
        the returned Obs has collided=True. There is no sliding and no
        damage model; back up and steer away.

        Returns Obs. Per-step draw order (frozen, see rng.py): odometry
        normal(3), then lidar normal(n_beams) + uniform(n_beams).
        """
        prev = self._pose
        new_pose = dynamics.step(prev, control, self.car_params, self.dt)
        collided = not self.map.is_free(new_pose[0], new_pose[1],
                                        radius=self.car_params.radius)
        if collided:
            new_pose = prev.copy()
        self._pose = new_pose
        self._trace.append(new_pose.copy())
        self._t += self.dt

        odom = self.odometry.measure(prev, new_pose)
        scan = self.lidar.scan(self.map, new_pose)
        return Obs(odom=odom, scan=scan, dt=self.dt, t=self._t,
                   collided=collided)

    # ------------------------------------------------------------- camera

    def render_camera(self):
        """Render one camera frame from the CURRENT true pose, on
        demand (frames are never rendered per step, so HW1 through HW5
        never pay for rendering).

        Returns np.ndarray (height, width, 3) RGB uint8, default 64x64.
        Each call consumes exactly one normal(size=2) draw [brightness,
        hue jitter] from the camera stream and nothing from any other
        stream, so rendering never perturbs lidar, odometry, or
        init-pose replay. The k-th render after a reset is a
        deterministic function of (seed, pose), bit-for-bit.
        """
        return self.camera.render(self.map, self._pose)

    # ------------------------------------------------------- ground truth

    @property
    def ground_truth_pose(self):
        """(3,) [x, y, theta] true pose. FOR PLOTTING AND AUTOGRADING
        ONLY: estimators that read this score zero."""
        return self._pose.copy()

    @property
    def ground_truth_trace(self):
        """(T, 3) array of every true pose since reset, including the
        initial pose. FOR PLOTTING AND AUTOGRADING ONLY."""
        return np.array(self._trace)

    # ------------------------------------------------------- extra streams

    def spawn_generator(self):
        """Spawn one more Generator off this episode's SeedSequence, for
        components outside the simulator (e.g. HW5's particle filter), so
        all course randomness descends from the one seed. Call order
        matters: the k-th call after a reset always returns the same
        stream. Does not disturb the four fixed component streams."""
        child = self._root_seq.spawn(1)[0]
        return np.random.Generator(np.random.PCG64(child))
