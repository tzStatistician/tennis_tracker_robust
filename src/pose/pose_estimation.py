from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

COCO17 = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

BONES = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
]

@dataclass
class PosePrediction:
    player_id: int
    label: str
    bbox: list[float]
    bbox_conf: float
    keypoints: Dict[str, Tuple[float, float, float]]


class PoseEstimator:
    def __init__(self, backend: str = 'ultralytics', weights: str = 'yolov8x-pose.pt', conf: float = 0.25):
        self.backend = backend
        self.weights = weights
        self.conf = conf
        self.model = None
        if backend == 'ultralytics':
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights)
            except Exception as e:
                print(f'[WARN] Could not initialize Ultralytics pose model: {e}. Pose disabled.')
                self.backend = 'none'

    def predict(self, frame) -> List[PosePrediction]:
        if self.backend == 'none' or self.model is None:
            return []
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        if not results:
            return []
        r = results[0]
        if r.boxes is None or r.keypoints is None:
            return []
        poses: List[PosePrediction] = []
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else [0.0] * len(xyxy)
        kxy = r.keypoints.xy.detach().cpu().numpy()
        kconf = r.keypoints.conf.detach().cpu().numpy() if r.keypoints.conf is not None else None
        order = sorted(range(len(xyxy)), key=lambda i: xyxy[i][3], reverse=True)[:2]
        for rank, i in enumerate(order):
            kp = {}
            for j, name in enumerate(COCO17):
                c = float(kconf[i][j]) if kconf is not None else 1.0
                kp[name] = (float(kxy[i][j][0]), float(kxy[i][j][1]), c)
            poses.append(PosePrediction(
                player_id=rank + 1,
                label='near_player' if rank == 0 else 'far_player',
                bbox=xyxy[i].astype(float).tolist(),
                bbox_conf=float(confs[i]),
                keypoints=kp,
            ))
        return poses
