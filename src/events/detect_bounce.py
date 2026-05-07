from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from src.ball.detect_ball import BallDetection


@dataclass
class BounceEvent:
    event_type: str
    frame_idx: int
    timestamp_sec: float
    player_id: Optional[int]
    label: Optional[str]
    ball_x: Optional[float]
    ball_y: Optional[float]
    confidence: float
    detail: Dict
    bounce_type: str = 'ground'  # 'ground' | 'net'


class BounceDetector:
    def __init__(self, min_frame_gap: int = 10, min_vertical_velocity_change: float = 2.0,
                 enabled: bool = True, frame_height: int = 1080,
                 net_region_y_ratio: float = 0.5, toss_history_len: int = 8,
                 toss_vy_threshold: float = -0.3):
        self.enabled = enabled
        self.min_frame_gap = min_frame_gap
        self.min_vy_change = min_vertical_velocity_change
        self.frame_height = frame_height
        self.net_region_y_max = frame_height * net_region_y_ratio
        self.toss_history_len = toss_history_len
        self.toss_vy_threshold = toss_vy_threshold
        self.history: List[tuple] = []
        self.last_bounce_frame = -10**9
        self.max_history = 5
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

    def update(self, frame_idx: int, timestamp_sec: float,
               ball: Optional[BallDetection]) -> Optional[BounceEvent]:
        if not self.enabled or ball is None:
            self.history = []
            self._update_history(ball)
            return None
        if ball.is_interpolated:
            self._update_history(ball)
            return None

        self._update_history(ball)
        self.history.append((frame_idx, ball.y))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        if len(self.history) < 3:
            return None

        ys = [y for _, y in self.history]
        mid = len(ys) // 2
        d1 = ys[mid] - ys[0]
        d2 = ys[-1] - ys[mid]

        direction_change = d1 > 1.0 and d2 < -1.0
        total_change = abs(d1) + abs(d2)

        if not direction_change or total_change < self.min_vy_change:
            return None

        if frame_idx - self.last_bounce_frame < self.min_frame_gap:
            return None

        # Suppress bounces during serve toss: toss peak bounces happen high in frame
        # (small y) while the ball is moving slowly upward then downward
        is_toss = self._is_toss_motion()
        if is_toss and ball.y < self.net_region_y_max:
            return None

        # Classify bounce type: net vs ground based on y position
        # In baseline camera view, net is in the upper portion of the frame
        if ball.y < self.net_region_y_max:
            bounce_type = 'net'
            conf = min(1.0, total_change / 10.0)  # net bounces have smaller velocity change
        else:
            bounce_type = 'ground'
            conf = min(1.0, total_change / 15.0)

        self.last_bounce_frame = frame_idx
        self.history = []
        return BounceEvent(
            event_type='bounce', frame_idx=frame_idx, timestamp_sec=timestamp_sec,
            player_id=None, label=None, ball_x=ball.x, ball_y=ball.y,
            confidence=conf,
            detail={'d1_px': float(d1), 'd2_px': float(d2),
                    'total_change_px': float(total_change), 'is_toss': is_toss},
            bounce_type=bounce_type
        )
