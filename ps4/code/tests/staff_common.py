"""Helpers for the released public tests."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

HW_DIR = os.path.dirname(HERE)

DATA_DIR = os.path.join(HW_DIR, "data")

def conv_refs():
    return np.load(os.path.join(DATA_DIR, "conv_public.npz"))

