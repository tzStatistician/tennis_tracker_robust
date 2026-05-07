from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.ball.detect_ball import BallDetection
from src.pose.pose_estimation import PosePrediction
from src.utils.geometry import euclidean


@dataclass
class HitEvent:
    event_type: str
    frame_idx: int
    timestamp_sec: float
    player_id: Optional[int]
    label: Optional[str]
    ball_x: Optional[float]
    ball_y: Optional[float]
    confidence: float
    detail: Dict


class HitDetector:
    def __init__(self, wrist_ball_distance_px: float = 90, min_frame_gap: int = 10,
                 velocity_change_weight: float = 0.4, enabled: bool = True,
                 toss_history_len: int = 8, toss_vy_threshold: float = -0.5,
                 frame_height: int = 1080, toss_region_y_ratio: float = 0.5):
        self.enabled = enabled
        self.distance_thr = wrist_ball_distance_px
        self.min_frame_gap = min_frame_gap
        self.velocity_change_weight = velocity_change_weight
        self.toss_history_len = toss_history_len
        self.toss_vy_threshold = toss_vy_threshold
        self.frame_height = frame_height
        self.toss_region_y_max = frame_height * toss_region_y_ratio
        self.last_hit_frame = -10**9
        self.prev_ball = None
        self.prev_velocity = None
        self.ball_y_history: List[float] = []

    def _is_toss_motion(self) -> bool:
        """Check if ball was moving upward (toss pattern) in recent frames."""
        if len(self.ball_y_history) < 4:
            return False
        recent = self.ball_y_history[-self.toss_history_len:]
        vy = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        up_count = sum(1 for v in vy if v < self.toss_vy_threshold)
        return up_count >= len(vy) * 0.5 and len(vy) >= 3

    def _update_history(self, ball: Optional[BallDetection]) -> None:
        if ball is not None:
            self.ball_y_history.append(ball.y)
            if len(self.ball_y_history) > self.toss_history_len * 2:
                self.ball_y_history = self.ball_y_history[-self.toss_history_len * 2:]

    def update(self, frame_idx: int, timestamp_sec: float, ball: Optional[BallDetection],
               poses: List[PosePrediction]) -> Optional[HitEvent]:
        if not self.enabled or ball is None or not poses:
            self._update_velocity(ball)
            self._update_history(ball)
            return None

        is_toss = self._is_toss_motion()

        best = None
        for pose in poses:
            for wrist in ('left_wrist', 'right_wrist'):
                if wrist not in pose.keypoints:
                    continue
                wx, wy, wc = pose.keypoints[wrist]
                if wc < 0.15:
                    continue
                d = euclidean((ball.x, ball.y), (wx, wy))
                score = max(0.0, 1.0 - d / max(self.distance_thr, 1e-6))
                if best is None or score > best[0]:
                    best = (score, d, wrist, pose)

        vchange = self._velocity_change(ball)
        self._update_velocity(ball)
        self._update_history(ball)

        if best is None:
            return None

        prox_score, dist, wrist, pose = best
        combined = float(min(1.0, prox_score + self.velocity_change_weight * min(1.0, vchange / 30.0)))

        # During toss: suppress hits from near_player (server).
        # Toss-release proximity looks like a hit but has low velocity change.
        # Only allow hits during toss if confidence is very high (real contact).
        if is_toss and ball.y < self.toss_region_y_max:
            if pose.label == 'near_player':
                # Real serve contact: ball is coming down fast after toss peak,
                # so velocity change is very high. Toss release: low vchange.
                if combined < 0.75 or vchange < 15.0:
                    return None
            # Far player during toss: still allow but require higher bar
            elif combined < 0.65:
                return None

        if combined >= 0.55 and frame_idx - self.last_hit_frame >= self.min_frame_gap:
            self.last_hit_frame = frame_idx
            return HitEvent(
                event_type='hit', frame_idx=frame_idx, timestamp_sec=timestamp_sec,
                player_id=pose.player_id, label=pose.label, ball_x=ball.x, ball_y=ball.y,
                confidence=combined,
                detail={'nearest_joint': wrist, 'distance_to_wrist_px': float(dist),
                        'velocity_change_px': float(vchange), 'is_toss': is_toss}
            )
        return None

    def _velocity_change(self, ball: Optional[BallDetection]) -> float:
        if ball is None or self.prev_ball is None or self.prev_velocity is None:
            return 0.0
        v = np.array([ball.x - self.prev_ball[0], ball.y - self.prev_ball[1]], dtype=float)
        return float(np.linalg.norm(v - self.prev_velocity))

    def _update_velocity(self, ball: Optional[BallDetection]) -> None:
        if ball is None:
            return
        cur = np.array([ball.x, ball.y], dtype=float)
        if self.prev_ball is not None:
            self.prev_velocity = cur - self.prev_ball
        self.prev_ball = cur
