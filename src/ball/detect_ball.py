from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

@dataclass
class BallDetection:
    x: float
    y: float
    conf: float
    radius: float = 3.0
    is_interpolated: bool = False


class BallDetector:
    def __init__(self, backend: str = 'color', weights: Optional[str] = None, conf: float = 0.15,
                 hsv_lower=None, hsv_upper=None, min_area: int = 4, max_area: int = 900):
        self.backend = backend
        self.weights = weights
        self.conf = conf
        self.hsv_lower = np.array(hsv_lower or [25, 40, 80], dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper or [90, 255, 255], dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.model = None
        if backend == 'ultralytics' and weights:
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights)
            except Exception as e:
                print(f'[WARN] Could not initialize Ultralytics ball detector: {e}. Falling back to color detector.')
                self.backend = 'color'

    def predict(self, frame) -> Optional[BallDetection]:
        if self.backend == 'none':
            return None
        if self.backend == 'ultralytics' and self.model is not None:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            candidates = []
            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    xyxy = b.xyxy[0].detach().cpu().numpy().astype(float)
                    c = float(b.conf[0].detach().cpu().item()) if b.conf is not None else 0.0
                    x = float((xyxy[0] + xyxy[2]) / 2)
                    y = float((xyxy[1] + xyxy[3]) / 2)
                    area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    candidates.append((c, area, x, y))
            if candidates:
                c, area, x, y = sorted(candidates, key=lambda z: z[0], reverse=True)[0]
                return BallDetection(x=x, y=y, conf=c, radius=max(2.0, min(10.0, area ** 0.5 / 2)))
            return None
        return self._predict_color(frame)

    def _predict_color(self, frame) -> Optional[BallDetection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.medianBlur(mask, 3)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            circularity = area / (np.pi * max(radius, 1e-6) ** 2)
            score = float(min(1.0, circularity) * min(1.0, area / 50.0))
            if best is None or score > best[0]:
                best = (score, x, y, radius)
        if best is None:
            return None
        score, x, y, radius = best
        return BallDetection(x=float(x), y=float(y), conf=score, radius=float(radius))
