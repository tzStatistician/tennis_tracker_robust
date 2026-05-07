from __future__ import annotations

import cv2
from src.pose.pose_estimation import BONES, PosePrediction


def draw_pose(frame, pose: PosePrediction, color=(255, 255, 255), draw_confidence: bool = False):
    for name, (x, y, c) in pose.keypoints.items():
        if c < 0.15:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        if draw_confidence:
            cv2.putText(frame, f'{name}:{c:.2f}', (int(x)+4, int(y)-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    for a, b in BONES:
        if a not in pose.keypoints or b not in pose.keypoints:
            continue
        x1, y1, c1 = pose.keypoints[a]
        x2, y2, c2 = pose.keypoints[b]
        if c1 < 0.15 or c2 < 0.15:
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame
