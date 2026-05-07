from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.ball.detect_ball import BallDetection
from src.pose.pose_estimation import PosePrediction
from src.visualize.draw_pose import draw_pose
from src.utils.geometry import bbox_center

# Color constants (BGR)
BALL_YELLOW = (0, 255, 255)
BALL_TRAIL_BRIGHT = (0, 230, 230)
BALL_TRAIL_DIM = (0, 80, 80)
NEAR_PLAYER = (255, 100, 50)
FAR_PLAYER = (50, 100, 255)
NEAR_TRAIL = (200, 80, 30)
FAR_TRAIL = (30, 80, 200)
HIT_COLOR = (0, 255, 0)          # green
BOUNCE_GROUND_COLOR = (0, 140, 255)  # orange
BOUNCE_NET_COLOR = (255, 0, 255)     # magenta
POSE_NEAR = (0, 255, 128)
POSE_FAR = (255, 128, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PLAYER_COLORS = {'near_player': NEAR_PLAYER, 'far_player': FAR_PLAYER}
TRAIL_COLORS = {'near_player': NEAR_TRAIL, 'far_player': FAR_TRAIL}
POSE_COLORS = {'near_player': POSE_NEAR, 'far_player': POSE_FAR}


class AugmentedVideoRenderer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        vcfg = cfg.get('visualization', {})
        self.ball_trail = deque(maxlen=int(vcfg.get('ball_trail_len', 25)))
        self.player_trails = defaultdict(lambda: deque(maxlen=int(vcfg.get('player_trail_len', 80))))
        self.recent_events = deque(maxlen=20)

    def draw(self, frame, frame_idx: int, timestamp_sec: float, ball: Optional[BallDetection],
             poses: List[PosePrediction], events: List[Dict], running_summary: Dict):
        vcfg = self.cfg.get('visualization', {})
        out = frame.copy()

        for e in events:
            self.recent_events.append(e)

        # Draw player bounding boxes and pose
        if vcfg.get('draw_players', True):
            for pose in poses:
                color = PLAYER_COLORS.get(pose.label, WHITE)
                pose_color = POSE_COLORS.get(pose.label, WHITE)

                x1, y1, x2, y2 = map(int, pose.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label_text = f'{pose.label.replace("_"," ").title()}'
                cv2.putText(out, label_text, (x1, max(15, y1 - 8)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Player movement trail
                center = bbox_center(pose.bbox)
                self.player_trails[pose.label].append(center)
                if vcfg.get('draw_player_trail', True):
                    pts = list(self.player_trails[pose.label])
                    trail_color = TRAIL_COLORS.get(pose.label, (128, 128, 128))
                    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                        alpha = 0.3 + 0.7 * i / max(1, len(pts))
                        c = tuple(int(v * alpha) for v in trail_color)
                        cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, 1)

                # Pose skeleton
                if vcfg.get('draw_pose', True):
                    draw_pose(out, pose, color=pose_color, draw_confidence=vcfg.get('draw_confidence', False))

        # Draw ball and trail
        if ball is not None and vcfg.get('draw_ball', True):
            self.ball_trail.append((ball.x, ball.y))
            if vcfg.get('draw_ball_trail', True):
                pts = list(self.ball_trail)
                for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                    t = i / max(1, len(pts))
                    c = tuple(int(BALL_TRAIL_DIM[j] + (BALL_TRAIL_BRIGHT[j] - BALL_TRAIL_DIM[j]) * t)
                             for j in range(3))
                    thickness = max(1, int(1 + 2 * t))
                    cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, thickness)
            # Ball circle
            if ball.is_interpolated:
                cv2.circle(out, (int(ball.x), int(ball.y)), max(6, int(ball.radius)),
                          (0, 180, 180), 2)
            else:
                cv2.circle(out, (int(ball.x), int(ball.y)), max(6, int(ball.radius)),
                          BALL_YELLOW, -1)
                cv2.circle(out, (int(ball.x), int(ball.y)), max(8, int(ball.radius) + 2),
                          BLACK, 1)

        # Draw events
        if vcfg.get('draw_events', True):
            for e in list(self.recent_events):
                age = frame_idx - int(e.get('frame_idx', frame_idx))
                if age < 0 or age > 30:
                    continue
                x, y = e.get('ball_x'), e.get('ball_y')
                if x is None or y is None:
                    continue
                event_type = e.get('event_type', '')
                bounce_type = e.get('bounce_type', 'ground')

                if event_type == 'hit':
                    color = HIT_COLOR
                    label = 'HIT'
                elif bounce_type == 'net':
                    color = BOUNCE_NET_COLOR
                    label = 'NET'
                else:
                    color = BOUNCE_GROUND_COLOR
                    label = 'BOUNCE'

                alpha = 1.0 - age / 30.0
                c = tuple(int(v * alpha) for v in color)
                cv2.circle(out, (int(x), int(y)), 18, c, 2)
                cv2.putText(out, label, (int(x) + 22, int(y) + 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        # Stats overlay
        if vcfg.get('draw_stats_overlay', True):
            self._draw_overlay(out, frame_idx, timestamp_sec, running_summary)

        return out

    def _draw_overlay(self, frame, frame_idx: int, timestamp_sec: float, summary: Dict):
        lines = [
            f'frame={frame_idx}  t={timestamp_sec:.2f}s',
            f"HITS={summary.get('num_hits', 0)}  BOUNCES={summary.get('num_bounces', 0)}",
            f"NEAR move={summary.get('near_player_total_movement_px', 0):.0f}px",
            f"FAR  move={summary.get('far_player_total_movement_px', 0):.0f}px",
        ]
        x, y0 = 12, 24
        for i, txt in enumerate(lines):
            y = y0 + i * 24
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 5)
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
