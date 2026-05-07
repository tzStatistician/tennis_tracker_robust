from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np

Point = Tuple[float, float]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:2], dtype=float) - np.asarray(b[:2], dtype=float)))


def apply_homography(point_xy: Sequence[float], H: Optional[np.ndarray]) -> Optional[Point]:
    if H is None:
        return None
    p = np.array([point_xy[0], point_xy[1], 1.0], dtype=float)
    q = H @ p
    if abs(q[2]) < 1e-9:
        return None
    return float(q[0] / q[2]), float(q[1] / q[2])


def bbox_center(bbox: Sequence[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def bbox_bottom_center(bbox: Sequence[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2), float(y2))
