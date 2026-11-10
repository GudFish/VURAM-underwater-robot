"""Shared Cartesian display geometry; image radii represent acoustic slant range."""
import math
import numpy as np


def cartesian_mask(height, width, min_range, max_range, fov_deg):
    forward = np.linspace(max_range, min_range, height)[:, None]
    span = max_range * math.sin(math.radians(fov_deg) * 0.5)
    lateral = np.linspace(span, -span, width)[None, :]
    radius = np.hypot(forward, lateral)
    return ((np.abs(np.arctan2(lateral, forward)) <= math.radians(fov_deg) * 0.5)
            & (radius >= min_range) & (radius <= max_range))
