"""coursesim: the course simulator (course simulator specification).

One package, pure numpy plus matplotlib, headless-first. Public API (the
F1 freeze surface):

    from coursesim import Simulator, Obs, Map, Cone, CarParams
    from coursesim.dynamics import step
    from coursesim.viz import MapView

viz is NOT imported here so that stepping code never pays for (or depends
on) matplotlib; import coursesim.viz explicitly when plotting.

Autograders assert a minimum __version__. Post-F1 changes are additive
only: fields append, names never change.
"""
from .dynamics import CarParams, step
from .sensors import Lidar, Odometry, OdometryNoise
from .simulator import Obs, Simulator
from .world import Cone, Map

__version__ = "0.1.0"

__all__ = ["Simulator", "Obs", "Map", "Cone", "CarParams", "step",
           "Lidar", "Odometry", "OdometryNoise", "__version__"]
