from __future__ import annotations

from typing import Optional
import numpy as np


def parse_homography(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError('court.homography must be a 3x3 matrix or null')
    return arr
