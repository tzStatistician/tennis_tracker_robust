from __future__ import annotations

from collections import deque
from typing import Optional
import numpy as np
from .detect_ball import BallDetection


class BallTrackSmoother:
    def __init__(self, window: int = 5, max_missing: int = 8):
        self.window = max(1, int(window))
        self.max_missing = max_missing
        self.points = deque(maxlen=self.window)
        self.last_det: Optional[BallDetection] = None
        self.missing_count = 0

    def update(self, det: Optional[BallDetection]) -> Optional[BallDetection]:
        if det is not None:
            self.missing_count = 0
            self.points.append((det.x, det.y))
            self.last_det = det
            arr = np.asarray(self.points, dtype=float)
            x, y = np.median(arr, axis=0)
            return BallDetection(x=float(x), y=float(y), conf=det.conf,
                                 radius=det.radius, is_interpolated=False)
        # Ball missing — try interpolation from recent trajectory
        if self.last_det is not None and self.missing_count < self.max_missing:
            self.missing_count += 1
            if len(self.points) >= 2:
                arr = np.asarray(self.points, dtype=float)
                # Simple linear extrapolation from last 2 smoothed positions
                dx = arr[-1][0] - arr[-2][0]
                dy = arr[-1][1] - arr[-2][1]
                x = arr[-1][0] + dx * self.missing_count
                y = arr[-1][1] + dy * self.missing_count
            else:
                x, y = self.last_det.x, self.last_det.y
            return BallDetection(x=float(x), y=float(y), conf=self.last_det.conf * 0.6,
                                 radius=self.last_det.radius, is_interpolated=True)
        self.last_det = None
        self.missing_count = 0
        return None
