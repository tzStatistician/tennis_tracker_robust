from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class PlayerDetection:
    player_id: int
    bbox: list[float]
    conf: float
    label: str = 'player'


class PlayerDetector:
    def __init__(self, backend: str = 'ultralytics', weights: str = 'yolov8x.pt', conf: float = 0.35,
                 classes: Optional[list[int]] = None, max_track_age: int = 30, min_movement_for_player: float = 20.0):
        self.backend = backend
        self.weights = weights
        self.conf = conf
        self.classes = classes or [0]
        self.model = None
        self.max_track_age = max_track_age
        self.min_movement = min_movement_for_player
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        if backend == 'ultralytics':
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights)
            except Exception as e:
                print(f'[WARN] Could not initialize player detector: {e}. Disabled.')
                self.backend = 'none'

    def predict(self, frame) -> List[PlayerDetection]:
        if self.backend == 'none' or self.model is None:
            return []
        results = self.model.predict(frame, conf=self.conf, classes=self.classes, verbose=False)
        raw_dets: list[tuple[float, float, float, float, float]] = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                c = float(b.conf[0].detach().cpu().item()) if b.conf is not None else 0.0
                raw_dets.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], c))

        raw_dets.sort(key=lambda d: d[3], reverse=True)

        matched_ids = set()
        active_tracks = []
        for det in raw_dets:
            x1, y1, x2, y2, c = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in matched_ids:
                    continue
                tx1, ty1, tx2, ty2 = t['bbox']
                iox1 = max(x1, tx1); ioy1 = max(y1, ty1)
                iox2 = min(x2, tx2); ioy2 = min(y2, ty2)
                iw = max(0, iox2 - iox1); ih = max(0, ioy2 - ioy1)
                inter = iw * ih
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (tx2 - tx1) * (ty2 - ty1)
                iou = inter / max(1e-6, area_a + area_b - inter)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is not None and best_iou > 0.25:
                t = self.tracks[best_id]
                t['bbox'] = (x1, y1, x2, y2, c)
                t['age'] = 0
                t['hits'] += 1
                prev_cx = t.get('center', (cx, cy))
                dist = np.sqrt((cx - prev_cx[0])**2 + (cy - prev_cx[1])**2)
                t['center'] = (cx, cy)
                t['total_movement'] += dist if dist > 2 else 0
                matched_ids.add(best_id)
                active_tracks.append(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': (x1, y1, x2, y2, c),
                    'center': (cx, cy),
                    'age': 0, 'hits': 1,
                    'total_movement': 0.0,
                }
                matched_ids.add(tid)
                active_tracks.append(tid)

        # Age unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched_ids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_track_age:
                    del self.tracks[tid]

        # Select top-2 players by total movement
        candidates = [(tid, t) for tid, t in self.tracks.items()
                      if t['age'] < 10 and t['hits'] >= 3 and t['total_movement'] > self.min_movement]
        if len(candidates) < 2:
            candidates = [(tid, t) for tid, t in self.tracks.items()
                          if t['age'] < 15 and t['hits'] >= 1]
        candidates.sort(key=lambda x: x[1]['total_movement'], reverse=True)
        selected = candidates[:2]

        detections: List[PlayerDetection] = []
        for rank, (tid, t) in enumerate(selected):
            x1, y1, x2, y2, c = t['bbox']
            detections.append(PlayerDetection(
                player_id=tid,
                bbox=[x1, y1, x2, y2],
                conf=c,
                label='near_player' if rank == 0 else 'far_player'
            ))
        return detections
