"""Matplotlib rendering: composable MapView layers and teleoperation.

MapView draws the map (obstacles, cones, start, goal) and then a stack of
composable overlay layers sharing one interface: each layer is an object
with draw(ax), added through the add_* methods and drawn in add order.
The same layer system supports path, value, tree, and particle visualizations,
so there are no assignment-specific rendering branches.

save(path) is headless (Agg); show() opens an interactive window. All
world coordinates are meters; grid overlays (heatmap, grid values) may use
any (h, w) shape and are stretched over the map extent, row 0 at the top
(maximum y), matching world.py's grid convention.

Teleop drives a Simulator from the keyboard. It also
runs scripted and headless: feed key names to step_key() and never call
run(); the smoke test does exactly that.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# Cone color names -> matplotlib colors (anything unknown passes through).
_CONE_COLORS = {"red": "tab:red", "blue": "tab:blue", "yellow": "gold",
                "green": "tab:green", "orange": "tab:orange"}


class _Layer:
    """One overlay: a draw(ax) callable plus its style kwargs. All add_*
    methods build these; MapView.render draws them in add order."""

    def __init__(self, draw_fn):
        self._draw_fn = draw_fn

    def draw(self, ax):
        self._draw_fn(ax)


class MapView:
    """Composable map figure. Construct with a world.Map, add overlay
    layers, then save(path) headless or show() interactive.

    Every add_* method appends a layer and returns it, so callers that
    animate can keep handles. Coordinates are world-frame meters unless
    the method says otherwise.
    """

    def __init__(self, map, figsize=None, show_cones=True,
                 show_start_goal=True):
        self.map = map
        self.figsize = figsize
        self.show_cones = show_cones
        self.show_start_goal = show_start_goal
        self._layers = []

    # ------------------------------------------------------------ layers

    def _add(self, draw_fn):
        layer = _Layer(draw_fn)
        self._layers.append(layer)
        return layer

    def add_path(self, points, color="tab:blue", linewidth=2.0, label=None):
        """Polyline. points: (M, 2) or (M, 3) world [x, y(, theta)]."""
        pts = np.asarray(points, dtype=float)

        def draw(ax):
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth,
                    label=label, zorder=4)
        return self._add(draw)

    def add_trace(self, trace, color="tab:orange", linewidth=1.5,
                  label=None):
        """Driven trajectory, e.g. sim.ground_truth_trace: (T, 2) or
        (T, 3) world poses. Same drawing as add_path, dashed."""
        pts = np.asarray(trace, dtype=float)

        def draw(ax):
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth,
                    linestyle="--", label=label, zorder=4)
        return self._add(draw)

    def add_tree(self, edges, color="0.45", linewidth=0.8):
        """Search/RRT tree. edges: sequence of polylines, each (K, 2)
        world points (K=2 for straight edges; Dubins arcs pass K>2), or
        one (E, 2, 2) array of segments."""
        polys = [np.asarray(e, dtype=float)[:, :2] for e in edges]

        def draw(ax):
            ax.add_collection(LineCollection(polys, colors=color,
                                             linewidths=linewidth, zorder=3))
        return self._add(draw)

    def add_particles(self, states, weights=None, color="tab:purple",
                      size=4.0):
        """Particle cloud. states: (N, 2) or (N, 3) world poses. weights:
        optional (N,), scales marker sizes (normalized to mean size)."""
        pts = np.asarray(states, dtype=float)
        s = np.full(len(pts), size)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            s = size * len(w) * w / max(w.sum(), 1e-300)

        def draw(ax):
            ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, alpha=0.6,
                       linewidths=0, zorder=5)
        return self._add(draw)

    def add_heatmap(self, values, cmap="viridis", alpha=0.6, vmin=None,
                    vmax=None):
        """Scalar field over the map. values: any (h, w) array stretched
        over the map extent, row 0 at the TOP (max y). NaN cells are
        transparent (use NaN for walls)."""
        vals = np.ma.masked_invalid(np.asarray(values, dtype=float))

        def draw(ax):
            ax.imshow(vals, extent=self.map.extent, origin="upper",
                      cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                      interpolation="nearest", zorder=2)
        return self._add(draw)

    def add_grid_values(self, values, fmt="{:.1f}", color="black",
                        fontsize=8):
        """Per-cell text (lecture VI/Q tables; keep grids small, this
        draws one text artist per cell). values: (h, w) stretched over the
        map extent like add_heatmap; NaN cells are skipped."""
        vals = np.asarray(values, dtype=float)

        def draw(ax):
            h, w = vals.shape
            x0, x1, y0, y1 = self.map.extent
            cw, ch = (x1 - x0) / w, (y1 - y0) / h
            for r in range(h):
                for c in range(w):
                    if np.isnan(vals[r, c]):
                        continue
                    ax.text(x0 + (c + 0.5) * cw, y1 - (r + 0.5) * ch,
                            fmt.format(vals[r, c]), ha="center",
                            va="center", color=color, fontsize=fontsize,
                            zorder=6)
        return self._add(draw)

    def add_car(self, pose, color="tab:red"):
        """Car marker: footprint circle plus heading tick. pose: (3,)
        [x, y, theta] world (m, m, rad). Radius = CarParams().radius."""
        from .dynamics import CarParams
        r = CarParams().radius
        p = np.asarray(pose, dtype=float)

        def draw(ax):
            ax.add_patch(plt.Circle((p[0], p[1]), r, facecolor=color,
                                    edgecolor="black", alpha=0.8, zorder=6))
            ax.plot([p[0], p[0] + 2 * r * np.cos(p[2])],
                    [p[1], p[1] + 2 * r * np.sin(p[2])],
                    color="black", linewidth=1.5, zorder=7)
        return self._add(draw)

    # --------------------------------------------------------- rendering

    def render(self):
        """Draw everything onto a fresh figure. Returns (fig, ax). save
        and show call this; call it directly to postprocess axes."""
        fig, ax = plt.subplots(figsize=self.figsize)
        m = self.map
        ax.imshow(m.occupancy, extent=m.extent, origin="upper",
                  cmap="gray_r", vmin=0, vmax=1, interpolation="nearest",
                  zorder=1)
        if self.show_cones:
            for cone in m.cones:
                ax.plot(cone.x, cone.y, marker="^", markersize=8,
                        color=_CONE_COLORS.get(cone.color, cone.color),
                        markeredgecolor="black", zorder=5)
        if self.show_start_goal:
            ax.plot(*m.start[:2], marker="o", markersize=9,
                    color="tab:green", markeredgecolor="black", zorder=5)
            ax.plot(*m.goal[:2], marker="*", markersize=14,
                    color="tab:red", markeredgecolor="black", zorder=5)
        for layer in self._layers:
            layer.draw(ax)
        ax.set_xlim(m.extent[0], m.extent[1])
        ax.set_ylim(m.extent[2], m.extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        return fig, ax

    def save(self, path, dpi=150):
        """Render and write an image file. Headless-safe: works with the
        Agg backend and no display."""
        fig, _ = self.render()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def show(self):
        """Render and open an interactive window (requires a display)."""
        self.render()
        plt.show()


class Teleop:
    """Keyboard teleoperation of a Simulator (HW1's drive-around).

    Key map (arrow keys or wasd):
      up / w      speed +0.25 m/s        down / s    speed -0.25 m/s
      left / a    steer +0.05 rad        right / d   steer -0.05 rad
      space / x   stop (zero speed and steering)
      q           quit the interactive loop

    Commands are clamped to the car's limits and HELD between keys; each
    tick steps the simulator once with the current (v, steer).

    Two modes share the same key handling:
      run()             interactive matplotlib window, one sim step per
                        timer tick (dt seconds each).
      step_key(key)     scripted/headless: apply one key (or None to just
                        hold), step once, return the Obs. Autograder smoke
                        tests drive this.
    """

    def __init__(self, sim):
        self.sim = sim
        self.v = 0.0
        self.steer = 0.0
        self._quit = False

    def handle_key(self, key):
        """Update the held (v, steer) command from one key name."""
        p = self.sim.car_params
        if key in ("up", "w"):
            self.v += 0.25
        elif key in ("down", "s"):
            self.v -= 0.25
        elif key in ("left", "a"):
            self.steer += 0.05
        elif key in ("right", "d"):
            self.steer -= 0.05
        elif key in (" ", "space", "x"):
            self.v, self.steer = 0.0, 0.0
        elif key == "q":
            self._quit = True
        self.v = float(np.clip(self.v, -p.max_speed, p.max_speed))
        self.steer = float(np.clip(self.steer, -p.max_steer, p.max_steer))

    def step_key(self, key=None):
        """Scripted mode: apply one key (None = hold current command),
        advance the sim one step, return the Obs."""
        if key is not None:
            self.handle_key(key)
        return self.sim.step((self.v, self.steer))

    def run(self, save_on_quit=None):
        """Interactive loop. Opens a window, steps the sim every dt
        seconds, redraws the car and trace. 'q' ends the loop; if
        save_on_quit is a path, the final view (with the driven trace) is
        saved there. Requires an interactive matplotlib backend."""
        view = MapView(self.sim.map)
        fig, ax = view.render()
        ax.set_title("teleop: arrows/wasd drive, space stops, q quits")
        car_dot, = ax.plot([], [], marker="o", markersize=10,
                           color="tab:red", zorder=6)
        trace_line, = ax.plot([], [], color="tab:orange", linewidth=1.5,
                              zorder=5)

        def on_key(event):
            self.handle_key(event.key)

        def tick():
            if self._quit:
                timer.stop()
                plt.close(fig)
                return
            self.sim.step((self.v, self.steer))
            pose = self.sim.ground_truth_pose
            trace = self.sim.ground_truth_trace
            car_dot.set_data([pose[0]], [pose[1]])
            trace_line.set_data(trace[:, 0], trace[:, 1])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)
        timer = fig.canvas.new_timer(interval=int(self.sim.dt * 1000))
        timer.add_callback(tick)
        timer.start()
        plt.show()
        if save_on_quit is not None:
            final = MapView(self.sim.map)
            final.add_trace(self.sim.ground_truth_trace)
            final.save(save_on_quit)
