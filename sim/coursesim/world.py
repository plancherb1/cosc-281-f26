"""Map: occupancy world, load/save, collision, and distance transform.

A Map is a YAML metadata file plus a grayscale PNG occupancy image. This
format is shared by provided and student-generated occupancy maps; Map.save
and Map.load therefore round-trip the same representation.

File format (frozen):
  <name>.yaml
    resolution: 0.05            # meters per pixel
    image: <name>.png           # occupancy raster, same directory
    origin: [0.0, 0.0]          # world (x, y) of the grid's LOWER-LEFT corner
    start: [1.5, 1.5, 0.0]      # [x, y, theta] pose (m, m, rad)
    goal: [13.5, 13.5, 0.0]     # [x, y, theta] pose (m, m, rad)
    cones:                      # visual landmarks for the future camera;
      - {x: 3.0, y: 4.0, color: red}   # NOT obstacles, lidar ignores them
  <name>.png
    8-bit grayscale. Pixel value 0 = OCCUPIED, 255 = free; the load
    threshold is value < 128 -> occupied. Image row 0 (top of the PNG) is
    the row of MAXIMUM world y.

Grid convention: a Cell is `(row, col)`, with row 0 at top and col 0 at
left. Cell (row, col) covers the world square
  x in [origin_x + col*res,          origin_x + (col+1)*res)
  y in [origin_y + (H-1-row)*res,    origin_y + (H-row)*res)
and grid_to_world returns cell CENTERS. All distances are measured between
cell centers, so geometry is accurate to about one resolution.
"""
import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from .dynamics import CarParams

# PNG pixel values below this threshold load as occupied.
_OCCUPIED_THRESH = 128


@dataclass
class Cone:
    """A colored cone landmark. x, y in meters (world frame); color is a
    lowercase name string (red, blue, yellow, green). Cones are visual
    landmarks for the future camera module only; they are not obstacles
    and the lidar does not see them."""
    x: float
    y: float
    color: str


class Map:
    """Occupancy-grid world with collision queries and a precomputed
    distance transform.

    Attributes (read-only by convention):
      occupancy      np.ndarray (H, W) bool, True = OCCUPIED
      resolution     meters per pixel
      origin         np.ndarray (2,), world (x, y) of the lower-left corner
      start          np.ndarray (3,) [x, y, theta] start pose (m, m, rad)
      goal           np.ndarray (3,) [x, y, theta] goal pose (m, m, rad)
      cones          list[Cone]
      distance_field np.ndarray (H, W) float, METERS from each cell center
                     to the nearest occupied cell center (0 inside
                     obstacles). Computed once at construction with
                     scipy.ndimage.distance_transform_edt. This is the
                     likelihood field used by range-sensor models.
      shape          (H, W)
    """

    def __init__(self, occupancy, resolution, origin=(0.0, 0.0),
                 start=(0.0, 0.0, 0.0), goal=(0.0, 0.0, 0.0), cones=None):
        """Build a Map from an in-memory grid. Generated maps can be saved
        with save(); provided course maps are read with load().

        occupancy: (H, W) bool array-like, True = occupied.
        resolution: meters per pixel. origin: world (x, y) of the
        lower-left corner. start, goal: [x, y, theta] poses. cones:
        list of Cone (or (x, y, color) tuples).
        """
        self.occupancy = np.asarray(occupancy, dtype=bool)
        if self.occupancy.ndim != 2:
            raise ValueError("occupancy must be a 2-D array")
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float).copy()
        self.start = np.asarray(start, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.cones = [c if isinstance(c, Cone) else Cone(*c)
                      for c in (cones or [])]
        if self.occupancy.any():
            self.distance_field = ndimage.distance_transform_edt(
                ~self.occupancy) * self.resolution
        else:
            self.distance_field = np.full(self.occupancy.shape, np.inf)

    # ------------------------------------------------------------ file I/O

    @classmethod
    def load(cls, path):
        """Load a map from its YAML file (the PNG is referenced inside).

        path: path to the .yaml file. Returns a Map. Raises FileNotFoundError
        if the YAML or its image is missing.
        """
        path = os.fspath(path)
        with open(path) as f:
            meta = yaml.safe_load(f)
        img_path = os.path.join(os.path.dirname(path), meta["image"])
        img = np.asarray(Image.open(img_path).convert("L"))
        occupancy = img < _OCCUPIED_THRESH
        cones = [Cone(float(c["x"]), float(c["y"]), str(c["color"]))
                 for c in meta.get("cones", [])]
        return cls(occupancy, meta["resolution"], meta["origin"],
                   meta["start"], meta["goal"], cones)

    def save(self, path):
        """Write the map as a YAML + PNG pair.

        path: destination .yaml path; the PNG lands next to it with the
        same basename (foo.yaml -> foo.png) and is referenced by basename
        so the pair stays relocatable. Round-trip contract: Map.load(path)
        after save(path) reproduces the grid and metadata exactly.
        """
        path = os.fspath(path)
        base, _ = os.path.splitext(path)
        png_path = base + ".png"
        img = np.where(self.occupancy, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(png_path)
        meta = {
            "resolution": self.resolution,
            "image": os.path.basename(png_path),
            "origin": [float(v) for v in self.origin],
            "start": [float(v) for v in self.start],
            "goal": [float(v) for v in self.goal],
            "cones": [{"x": float(c.x), "y": float(c.y), "color": c.color}
                      for c in self.cones],
        }
        with open(path, "w") as f:
            yaml.safe_dump(meta, f, default_flow_style=None, sort_keys=False)

    # ------------------------------------------------- coordinate transforms

    @property
    def shape(self):
        """(H, W) of the occupancy grid."""
        return self.occupancy.shape

    @property
    def extent(self):
        """(x_min, x_max, y_min, y_max) world bounds in meters. Matches
        matplotlib imshow's extent argument order."""
        h, w = self.occupancy.shape
        x0, y0 = self.origin
        return (x0, x0 + w * self.resolution, y0, y0 + h * self.resolution)

    def world_to_grid(self, x, y):
        """World (x, y) meters -> grid (row, col). Vectorized: scalars in,
        integer scalars out; arrays in, integer arrays out. No bounds
        clipping; callers that may query outside the map use is_free,
        which treats out-of-bounds as not free."""
        h = self.occupancy.shape[0]
        col = np.floor((np.asarray(x, dtype=float) - self.origin[0])
                       / self.resolution).astype(np.int64)
        row = (h - 1) - np.floor((np.asarray(y, dtype=float) - self.origin[1])
                                 / self.resolution).astype(np.int64)
        return row, col

    def grid_to_world(self, row, col):
        """Grid (row, col) -> world (x, y) meters at the CELL CENTER.
        Vectorized like world_to_grid."""
        h = self.occupancy.shape[0]
        x = self.origin[0] + (np.asarray(col, dtype=float) + 0.5) * self.resolution
        y = self.origin[1] + (h - np.asarray(row, dtype=float) - 0.5) * self.resolution
        return x, y

    # ------------------------------------------------------ collision queries

    def is_free(self, x, y, radius=0.0):
        """True where a disc of the given radius centered at world (x, y)
        clears all obstacles. radius=0.0 is a point query. Out-of-bounds
        points are NOT free. Vectorized: accepts scalars or arrays and
        returns bools of the same shape.

        The footprint check is distance_field > radius at the containing
        cell, so it inherits the cell-center accuracy of the transform
        (about one resolution).
        """
        row, col = self.world_to_grid(x, y)
        h, w = self.occupancy.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        rr, cc = np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)
        free = inside & (self.distance_field[rr, cc] > radius)
        if free.ndim == 0:
            return bool(free)
        return free

    def occupancy_grid(self, inflated=True, radius=None):
        """Public occupancy-array accessor used by grid planners.

        inflated=True returns the grid with obstacles grown by `radius`
        meters (default: CarParams().radius, the course car), so a point
        planner plans for the real footprint. inflated=False returns the
        raw grid. Returns a NEW (H, W) bool array, True = obstacle;
        mutating it does not touch the Map.
        """
        if not inflated:
            return self.occupancy.copy()
        if radius is None:
            radius = CarParams().radius
        return self.distance_field <= radius
