"""Helpers for the released public tests."""
import importlib.util

import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

HW_DIR = os.path.dirname(HERE)

ROOT = os.path.dirname(os.path.dirname(HW_DIR))

CORRIDOR_MAP = os.path.join(ROOT, "sim", "maps", "corridor.yaml")

STAFF_MODELS_PY = os.path.join(HW_DIR, "models.py")

_cache = {}

def _load_by_path(name, path, register=False):
    """Import a module from an explicit file path (never from the
    submission). register=True also places it in sys.modules so the
    module's own absolute imports (e.g. astar's `from search_utils
    import ...`) resolve to the staff copy."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def staff_models():
    """The COURSE COPY of models.py (starter/), immune to submission
    edits."""
    if "models" not in _cache:
        _cache["models"] = _load_by_path("ps3_staff_models",
                                         STAFF_MODELS_PY)
    return _cache["models"]

def corridor():
    """The corridor Map, loaded once (read-only by convention)."""
    if "corridor" not in _cache:
        from coursesim import Map
        _cache["corridor"] = Map.load(CORRIDOR_MAP)
    return _cache["corridor"]

def control_experiment():
    """Use the provided experiment, not an editable submission runner."""
    if "control_experiment" not in _cache:
        _cache["control_experiment"] = _load_by_path(
            "ps3_control_experiment",
            os.path.join(HW_DIR, "control_experiment.py"))
    return _cache["control_experiment"]

