from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np


def _distance(a, b) -> float:
    if a is None or b is None:
        return 0.0
    if any(np.isnan(v) for v in (*a, *b)):
        return 0.0
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def summarize_point(frame_rows: List[Dict], events: List[Dict], fps: float, point_id: str) -> Dict:
    total_frames = len(frame_rows)
    duration_sec = total_frames / max(fps, 1e-9)
    summary = {
        'point_id': point_id,
        'duration_sec': round(duration_sec, 3),
        'fps': fps,
        'total_frames': total_frames,
        'num_hits': sum(1 for e in events if e.get('event_type') == 'hit'),
        'num_bounces': sum(1 for e in events if e.get('event_type') == 'bounce'),
        'rally_length_proxy': sum(1 for e in events if e.get('event_type') == 'hit'),
    }
    for label in ['near_player', 'far_player']:
        centers = []
        for r in frame_rows:
            x = r.get(f'{label}_center_x_px')
            y = r.get(f'{label}_center_y_px')
            if x is not None and y is not None and not (np.isnan(x) or np.isnan(y)):
                centers.append((x, y))
        total_px = 0.0
        speeds = []
        for a, b in zip(centers[:-1], centers[1:]):
            d = _distance(a, b)
            if d < 1.5:  # ignore tiny pixel jitter
                d = 0.0
            total_px += d
            speeds.append(d * fps)
        summary[f'{label}_total_movement_px'] = round(total_px, 3)
        summary[f'{label}_avg_speed_pxps'] = round(float(np.mean(speeds)) if speeds else 0.0, 3)
        summary[f'{label}_max_speed_pxps'] = round(float(np.max(speeds)) if speeds else 0.0, 3)
        summary[f'{label}_num_hits'] = sum(1 for e in events if e.get('event_type') == 'hit' and e.get('label') == label)
    # ball trajectory length in pixels
    ball_pts = [(r.get('ball_x_px'), r.get('ball_y_px')) for r in frame_rows if r.get('ball_x_px') is not None]
    ball_len = sum(_distance(a, b) for a, b in zip(ball_pts[:-1], ball_pts[1:]))
    summary['ball_trajectory_length_px'] = round(ball_len, 3)
    return summary
