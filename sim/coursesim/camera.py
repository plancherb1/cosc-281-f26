"""Camera: tiny deterministic numpy software rasterizer.

Deliberately crude, deliberately deterministic. A pinhole camera at the
car pose renders the map's cones as colored filled triangles, walls as
gray boxes (one ray march per image column, Wolfenstein style), and a
ground/sky gradient behind everything. Output is 64x64 RGB uint8. No GL,
no display, and no photorealism. The fixed 64x64 output keeps perception
experiments fast and compact.

Randomness: the camera owns one child Generator (the reserved `camera`
stream, rng.py spawn index 3) and draws ONLY per-image photometric
jitter from it. Draw-order contract (frozen, mirrors sensors.py):

  render() always draws normal(size=2) [brightness, hue], even when both
  jitter scales are zero, so turning a jitter knob never shifts the
  stream. Nothing else in this module draws.

Geometry conventions (world frame matches dynamics.py):
  camera center at (pose x, pose y), height cam_height above the ground
  plane, optical axis along the car heading theta, horizon at the image
  vertical center. Column 0 is the LEFT edge of the image, so a landmark
  at positive bearing (left of the heading) projects left of center.
  Row 0 is the TOP. Units: meters and radians in, pixels out.
"""
import numpy as np

# Cone landmark colors, RGB uint8, matched to world.Cone.color names.
CONE_COLORS = {
    "red": (215, 60, 40),
    "yellow": (235, 200, 50),
    "blue": (50, 90, 210),
    "green": (60, 170, 80),
}

# Background gradient anchors, RGB.
_SKY_TOP = np.array([120, 170, 235], dtype=float)
_SKY_HORIZON = np.array([205, 222, 245], dtype=float)
_GROUND_HORIZON = np.array([105, 110, 100], dtype=float)
_GROUND_BOTTOM = np.array([160, 165, 155], dtype=float)

# Wall shading: gray value at zero distance and at the far clip.
_WALL_NEAR_GRAY = 130.0
_WALL_FAR_GRAY = 60.0


def _rgb_to_hsv(rgb):
    """Vectorized RGB -> HSV, both float arrays in [0, 1], shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    span = maxc - minc
    s = np.where(maxc > 0, span / np.maximum(maxc, 1e-12), 0.0)
    safe = np.maximum(span, 1e-12)
    rc = (maxc - r) / safe
    gc = (maxc - g) / safe
    bc = (maxc - b) / safe
    h = np.where(maxc == r, bc - gc,
                 np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(span > 0, (h / 6.0) % 1.0, 0.0)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv):
    """Vectorized HSV -> RGB, both float arrays in [0, 1], shape (..., 3)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    choices_r = np.stack([v, q, p, p, t, v])
    choices_g = np.stack([t, v, v, q, p, p])
    choices_b = np.stack([p, p, t, v, v, q])
    idx = i[None]
    r = np.take_along_axis(choices_r, idx, axis=0)[0]
    g = np.take_along_axis(choices_g, idx, axis=0)[0]
    b = np.take_along_axis(choices_b, idx, axis=0)[0]
    return np.stack([r, g, b], axis=-1)


class Camera:
    """Pinhole software rasterizer over a coursesim Map.

    Constructor arguments (SI units unless noted):
      width, height     output image size in pixels (default 64x64)
      fov_deg           horizontal field of view, degrees (default 60)
      cam_height        camera center above the ground plane, m
      cone_height       rendered cone height, m
      cone_radius       rendered cone base half-width, m
      wall_height       rendered wall top above the ground plane, m
      far               wall render range, m (columns that hit nothing
                        within `far` show open horizon)
      near              near clip, m; scene points closer are skipped
      brightness_std    std of the multiplicative brightness jitter
      hue_std           std of the additive hue jitter (hue in [0, 1))
      rng               np.random.Generator, the camera's ONLY
                        randomness source (required)
    """

    def __init__(self, width=64, height=64, fov_deg=60.0, cam_height=0.20,
                 cone_height=0.40, cone_radius=0.12, wall_height=0.50,
                 far=8.0, near=0.12, brightness_std=0.12, hue_std=0.035,
                 rng=None):
        if rng is None:
            raise ValueError("Camera requires an explicit rng Generator")
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.cam_height = float(cam_height)
        self.cone_height = float(cone_height)
        self.cone_radius = float(cone_radius)
        self.wall_height = float(wall_height)
        self.far = float(far)
        self.near = float(near)
        self.brightness_std = float(brightness_std)
        self.hue_std = float(hue_std)
        self.rng = rng
        # Focal length in pixels from the horizontal FOV; principal
        # point at the image center (pixel-center convention).
        self.focal_px = (self.width / 2.0) / np.tan(
            np.deg2rad(self.fov_deg) / 2.0)
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0
        # Per-column bearing offsets, radians, positive = left.
        cols = np.arange(self.width)
        self._col_bearings = np.arctan((self.cx - cols) / self.focal_px)

    # ------------------------------------------------------------ geometry

    def _wall_forward_distances(self, map, pose):
        """Forward-axis distance to the first occupied cell per image
        column, marching one ray per column (half-cell steps, like
        Lidar.scan). Returns (dist (W,), hit (W,) bool); dist is np.inf
        where nothing was hit within `far`. Consumes NO randomness."""
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        angles = theta + self._col_bearings                     # (W,)
        step = 0.5 * map.resolution
        ts = np.arange(1, int(self.far / step) + 1) * step      # (S,)
        xs = x + ts[:, None] * np.cos(angles)[None, :]          # (S, W)
        ys = y + ts[:, None] * np.sin(angles)[None, :]
        blocked = ~map.is_free(xs, ys)                          # (S, W)
        hit = blocked.any(axis=0)                               # (W,)
        first = np.argmax(blocked, axis=0)                      # (W,)
        along = np.where(hit, ts[first], np.inf)                # ray length
        # Forward-axis (perpendicular) distance, so wall boxes and cone
        # depths compare in the same metric with no fisheye.
        return along * np.cos(self._col_bearings), hit

    def _cone_camera_coords(self, pose, cone):
        """(forward, left) coordinates of a cone center in the camera
        frame, meters. forward > 0 means in front of the camera."""
        dx = cone.x - float(pose[0])
        dy = cone.y - float(pose[1])
        c, s = np.cos(float(pose[2])), np.sin(float(pose[2]))
        return c * dx + s * dy, -s * dx + c * dy

    def visible_cones(self, map, pose, max_range=4.5, min_px=5.0):
        """Ground-truth visibility labels for the datagen pipeline.

        A cone is visible when its center column lands inside the image,
        its forward distance is in [near, max_range], its apparent pixel
        height is at least min_px, and the wall ray through its center
        column does not hit an obstacle in front of it (occlusion).

        Returns a list of dicts sorted near to far, one per visible
        cone: {"color", "u" (col, float), "v_top", "v_base" (rows,
        float), "px_height", "distance" (forward m), "centroid"
        ((row, col) of the filled triangle's centroid)}. Consumes NO
        randomness; safe to call without touching the jitter stream.
        """
        wall_d, _ = self._wall_forward_distances(map, pose)
        out = []
        for cone in map.cones:
            fwd, left = self._cone_camera_coords(pose, cone)
            if not (self.near <= fwd <= max_range):
                continue
            u = self.cx - self.focal_px * left / fwd
            if not (0.0 <= u <= self.width - 1):
                continue
            px_height = self.focal_px * self.cone_height / fwd
            if px_height < min_px:
                continue
            col = int(round(u))
            if wall_d[col] <= fwd:
                continue
            v_top = self.cy - self.focal_px * (self.cone_height
                                               - self.cam_height) / fwd
            v_base = self.cy + self.focal_px * self.cam_height / fwd
            centroid = (v_top + (2.0 / 3.0) * (v_base - v_top), u)
            out.append({"color": cone.color, "u": u, "v_top": v_top,
                        "v_base": v_base, "px_height": px_height,
                        "distance": fwd, "centroid": centroid})
        out.sort(key=lambda d: d["distance"])
        return out

    # ------------------------------------------------------------ rendering

    def _background(self):
        """(H, W, 3) float ground/sky vertical gradient, horizon at cy."""
        rows = np.arange(self.height, dtype=float)
        img = np.empty((self.height, self.width, 3), dtype=float)
        sky = rows <= self.cy
        t_sky = (rows[sky] / max(self.cy, 1e-9))[:, None]
        img[sky] = ((1.0 - t_sky) * _SKY_TOP
                    + t_sky * _SKY_HORIZON)[:, None, :]
        gnd = ~sky
        t_gnd = ((rows[gnd] - self.cy)
                 / max(self.height - 1 - self.cy, 1e-9))[:, None]
        img[gnd] = ((1.0 - t_gnd) * _GROUND_HORIZON
                    + t_gnd * _GROUND_BOTTOM)[:, None, :]
        return img

    def render(self, map, pose):
        """Render one frame from a pose. pose: (3,) [x, y, theta].

        Returns np.ndarray (height, width, 3) RGB uint8. Consumes
        exactly one normal(size=2) draw from the camera stream
        [brightness factor, hue shift], drawn even when both jitter
        scales are zero (fixed-draw contract, rng.py).
        """
        img = self._background()
        rows = np.arange(self.height, dtype=float)[:, None]     # (H, 1)

        # Walls: gray vertical boxes, one per column, shaded by distance.
        wall_d, hit = self._wall_forward_distances(map, pose)
        with np.errstate(divide="ignore"):
            v_top = self.cy - self.focal_px * (self.wall_height
                                               - self.cam_height) / wall_d
            v_bot = self.cy + self.focal_px * self.cam_height / wall_d
        frac = np.clip(np.where(hit, wall_d, self.far) / self.far, 0.0, 1.0)
        gray = _WALL_NEAR_GRAY + (_WALL_FAR_GRAY - _WALL_NEAR_GRAY) * frac
        wall_mask = hit[None, :] & (rows >= v_top[None, :]) \
            & (rows <= v_bot[None, :])                          # (H, W)
        img = np.where(wall_mask[:, :, None],
                       gray[None, :, None] * np.ones(3), img)

        # Cones: filled triangles, painter's order far to near, occluded
        # per column against the wall depth.
        cols = np.arange(self.width, dtype=float)[None, :]      # (1, W)
        cone_depth = []
        for cone in map.cones:
            fwd, left = self._cone_camera_coords(pose, cone)
            if fwd >= self.near:
                cone_depth.append((fwd, left, cone))
        cone_depth.sort(key=lambda t: -t[0])
        for fwd, left, cone in cone_depth:
            u = self.cx - self.focal_px * left / fwd
            half_w = self.focal_px * self.cone_radius / fwd
            if u + half_w < 0 or u - half_w > self.width - 1:
                continue
            v_top_c = self.cy - self.focal_px * (self.cone_height
                                                 - self.cam_height) / fwd
            v_base_c = self.cy + self.focal_px * self.cam_height / fwd
            span = max(v_base_c - v_top_c, 1e-9)
            # Triangle: half-width grows linearly from apex to base.
            hw = half_w * np.clip((rows - v_top_c) / span, 0.0, None)
            tri = (rows >= v_top_c) & (rows <= v_base_c) \
                & (np.abs(cols - u) <= hw)                      # (H, W)
            unoccluded = fwd < np.where(hit, wall_d, np.inf)    # (W,)
            mask = tri & unoccluded[None, :]
            color = np.array(CONE_COLORS.get(cone.color,
                                             CONE_COLORS["red"]), dtype=float)
            img = np.where(mask[:, :, None], color[None, None, :], img)

        # Photometric jitter: ONE normal(2) draw per render, always.
        jit = self.rng.normal(0.0, 1.0, 2)
        brightness = float(np.clip(1.0 + self.brightness_std * jit[0],
                                   0.6, 1.4))
        hue_shift = float(np.clip(self.hue_std * jit[1], -0.09, 0.09))
        hsv = _rgb_to_hsv(img / 255.0)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
        hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0.0, 1.0)
        out = _hsv_to_rgb(hsv) * 255.0
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
