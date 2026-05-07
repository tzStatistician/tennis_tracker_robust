from __future__ import annotations

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4,5')

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_yaml, deep_update
from src.utils.io import ensure_dir, write_json
from src.utils.geometry import bbox_center, bbox_bottom_center
from src.video.reader import VideoReader
from src.video.writer import VideoWriter
from src.player.detect_player import PlayerDetector
from src.pose.pose_estimation import PoseEstimator, PosePrediction
from src.ball.detect_ball import BallDetector, BallDetection
from src.ball.track_ball import BallTrackSmoother
from src.events.detect_hit import HitDetector
from src.events.detect_bounce import BounceDetector
from src.stats.player_stats import summarize_point
from src.visualize.renderer import AugmentedVideoRenderer

SELECTED_KPTS = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def parse_args():
    p = argparse.ArgumentParser(description='Offline AI tennis point video analyzer')
    p.add_argument('--config', default='configs/default.yaml', help='Path to YAML config')
    p.add_argument('--video', required=True, help='Input point video path')
    p.add_argument('--output_root', default=None, help='Root output directory')
    p.add_argument('--point_id', default=None, help='Point ID. Defaults to video stem.')
    p.add_argument('--max_frames', type=int, default=None, help='Debug: process at most N frames')
    p.add_argument('--ball_backend', default=None, choices=['color', 'ultralytics', 'none'], help='Override ball backend')
    p.add_argument('--ball_weights', default=None, help='Path to custom ball detector weights')
    p.add_argument('--no_video', action='store_true', help='Disable augmented video writing')
    return p.parse_args()


def choose_pose_as_player_source(poses: List[PosePrediction]) -> List[PosePrediction]:
    # YOLO pose returns bbox and skeleton together. For v0.1, we use pose predictions as the main player records.
    return poses


def pose_anchor(pose: PosePrediction, mode: str) -> Optional[tuple[float, float]]:
    kp = pose.keypoints
    def ok(name):
        return name in kp and kp[name][2] >= 0.15
    if mode == 'ankles_midpoint' and ok('left_ankle') and ok('right_ankle'):
        return ((kp['left_ankle'][0] + kp['right_ankle'][0]) / 2, (kp['left_ankle'][1] + kp['right_ankle'][1]) / 2)
    if mode in ('ankles_midpoint', 'hips_midpoint') and ok('left_hip') and ok('right_hip'):
        return ((kp['left_hip'][0] + kp['right_hip'][0]) / 2, (kp['left_hip'][1] + kp['right_hip'][1]) / 2)
    return bbox_bottom_center(pose.bbox)


def build_frame_row(point_id: str, frame_idx: int, timestamp_sec: float, ball: Optional[BallDetection], poses: List[PosePrediction], frame_events: List[Dict], anchor_mode: str) -> Dict:
    row: Dict = {
        'point_id': point_id,
        'frame_idx': frame_idx,
        'timestamp_sec': timestamp_sec,
        'ball_x_px': ball.x if ball else np.nan,
        'ball_y_px': ball.y if ball else np.nan,
        'ball_conf': ball.conf if ball else np.nan,
        'ball_is_interpolated': bool(ball.is_interpolated) if ball else False,
        'is_hit': any(e.get('event_type') == 'hit' for e in frame_events),
        'is_bounce': any(e.get('event_type') == 'bounce' for e in frame_events),
        'bounce_type': next((e.get('bounce_type') for e in frame_events if e.get('event_type') == 'bounce'), None),
        'hit_player_id': next((e.get('player_id') for e in frame_events if e.get('event_type') == 'hit'), None),
        'hit_confidence': next((e.get('confidence') for e in frame_events if e.get('event_type') == 'hit'), np.nan),
        'bounce_confidence': next((e.get('confidence') for e in frame_events if e.get('event_type') == 'bounce'), np.nan),
    }
    for label in ['near_player', 'far_player']:
        pose = next((p for p in poses if p.label == label), None)
        if pose is None:
            row.update({
                f'{label}_bbox_x1': np.nan, f'{label}_bbox_y1': np.nan,
                f'{label}_bbox_x2': np.nan, f'{label}_bbox_y2': np.nan,
                f'{label}_center_x_px': np.nan, f'{label}_center_y_px': np.nan,
                f'{label}_conf': np.nan,
            })
            for k in SELECTED_KPTS:
                row[f'{label}_{k}_x'] = np.nan
                row[f'{label}_{k}_y'] = np.nan
                row[f'{label}_{k}_conf'] = np.nan
            continue
        x1, y1, x2, y2 = pose.bbox
        anchor = pose_anchor(pose, anchor_mode)
        row.update({
            f'{label}_bbox_x1': x1, f'{label}_bbox_y1': y1,
            f'{label}_bbox_x2': x2, f'{label}_bbox_y2': y2,
            f'{label}_center_x_px': anchor[0] if anchor else np.nan,
            f'{label}_center_y_px': anchor[1] if anchor else np.nan,
            f'{label}_conf': pose.bbox_conf,
        })
        for k in SELECTED_KPTS:
            x, y, c = pose.keypoints.get(k, (np.nan, np.nan, np.nan))
            row[f'{label}_{k}_x'] = x
            row[f'{label}_{k}_y'] = y
            row[f'{label}_{k}_conf'] = c
    return row


def event_to_dict(e) -> Dict:
    d = {
        'event_type': e.event_type,
        'frame_idx': e.frame_idx,
        'timestamp_sec': e.timestamp_sec,
        'player_id': e.player_id,
        'label': e.label,
        'ball_x': e.ball_x,
        'ball_y': e.ball_y,
        'confidence': e.confidence,
        **{f'detail_{k}': v for k, v in e.detail.items()},
    }
    if hasattr(e, 'bounce_type'):
        d['bounce_type'] = e.bounce_type
    return d


def run_analysis(cfg: Dict):
    video_path = Path(cfg['input']['video_path'])
    point_id = cfg['input'].get('point_id') or video_path.stem
    output_root = Path(cfg['output']['root_dir'])
    out_dir = ensure_dir(output_root / point_id)

    reader = VideoReader(video_path)
    fps = reader.info.fps
    out_fps = cfg.get('video', {}).get('output_fps') or fps
    writer = None
    if cfg['output'].get('save_augmented_video', True):
        writer = VideoWriter(out_dir / 'augmented_video.mp4', out_fps, reader.info.width, reader.info.height)

    mcfg = cfg.get('models', {})
    # Detector is initialized for future use, but pose model already supplies player bboxes in v0.1.
    _player_detector = PlayerDetector(**mcfg.get('player_detector', {}))
    pose_estimator = PoseEstimator(**mcfg.get('pose_estimator', {}))
    ball_cfg = mcfg.get('ball_detector', {})
    tracking_cfg = cfg.get('tracking', {})

    ball_detector = BallDetector(**ball_cfg)
    ball_smoother = BallTrackSmoother(
        window=tracking_cfg.get('ball_smoothing_window', 5),
        max_missing=tracking_cfg.get('max_missing_ball_frames', 8))
    hit_detector = HitDetector(**cfg.get('events', {}).get('hit', {}), frame_height=reader.info.height)
    bounce_detector = BounceDetector(**cfg.get('events', {}).get('bounce', {}), frame_height=reader.info.height)
    renderer = AugmentedVideoRenderer(cfg)

    frame_rows: List[Dict] = []
    events: List[Dict] = []
    every_n = int(cfg.get('video', {}).get('process_every_n_frames', 1))
    max_frames = cfg.get('video', {}).get('max_frames')

    pbar = tqdm(total=reader.info.total_frames if max_frames is None else min(max_frames, reader.info.total_frames), desc=f'Analyzing {point_id}')
    try:
        for frame_idx, timestamp_sec, frame in reader.frames(every_n=every_n, max_frames=max_frames):
            poses = pose_estimator.predict(frame)
            poses = choose_pose_as_player_source(poses)
            raw_ball = ball_detector.predict(frame)
            ball = ball_smoother.update(raw_ball)

            frame_events: List[Dict] = []
            he = hit_detector.update(frame_idx, timestamp_sec, ball, poses)
            if he is not None:
                d = event_to_dict(he)
                events.append(d); frame_events.append(d)
            be = bounce_detector.update(frame_idx, timestamp_sec, ball)
            if be is not None:
                d = event_to_dict(be)
                events.append(d); frame_events.append(d)

            row = build_frame_row(point_id, frame_idx, timestamp_sec, ball, poses, frame_events, cfg.get('tracking', {}).get('player_anchor', 'ankles_midpoint'))
            frame_rows.append(row)
            running_summary = summarize_point(frame_rows, events, fps=fps / max(every_n, 1), point_id=point_id)

            if writer is not None:
                annotated = renderer.draw(frame, frame_idx, timestamp_sec, ball, poses, frame_events, running_summary)
                writer.write(annotated)
            pbar.update(1)
    finally:
        pbar.close()
        reader.release()
        if writer is not None:
            writer.release()

    effective_fps = fps / max(every_n, 1)
    point_summary = summarize_point(frame_rows, events, effective_fps, point_id)
    frame_df = pd.DataFrame(frame_rows)
    events_df = pd.DataFrame(events)
    summary_df = pd.DataFrame([point_summary])

    if cfg['output'].get('save_frame_csv', True):
        frame_df.to_csv(out_dir / 'frame_report.csv', index=False)
    if cfg['output'].get('save_frame_json', True):
        frame_df.to_json(out_dir / 'frame_report.json', orient='records', indent=2)
    if cfg['output'].get('save_events_csv', True):
        events_df.to_csv(out_dir / 'events.csv', index=False)
    if cfg['output'].get('save_events_json', True):
        events_df.to_json(out_dir / 'events.json', orient='records', indent=2)
    if cfg['output'].get('save_point_summary_csv', True):
        summary_df.to_csv(out_dir / 'point_summary.csv', index=False)
    if cfg['output'].get('save_point_summary_json', True):
        write_json(point_summary, out_dir / 'point_summary.json')

    write_json({
        'point_id': point_id,
        'input_video': str(video_path),
        'output_dir': str(out_dir),
        'source_fps': fps,
        'effective_fps': effective_fps,
        'width': reader.info.width,
        'height': reader.info.height,
        'total_source_frames': reader.info.total_frames,
        'processed_frames': len(frame_rows),
        'outputs': {
            'augmented_video': str(out_dir / 'augmented_video.mp4') if writer is not None else None,
            'frame_report_csv': str(out_dir / 'frame_report.csv'),
            'events_csv': str(out_dir / 'events.csv'),
            'point_summary_json': str(out_dir / 'point_summary.json'),
        }
    }, out_dir / 'run_metadata.json')

    print(f'\nDone. Outputs saved to: {out_dir}')
    return out_dir


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    overrides = {'input': {'video_path': args.video}}
    if args.output_root is not None:
        overrides.setdefault('output', {})['root_dir'] = args.output_root
    if args.point_id is not None:
        overrides.setdefault('input', {})['point_id'] = args.point_id
    if args.max_frames is not None:
        overrides.setdefault('video', {})['max_frames'] = args.max_frames
    if args.no_video:
        overrides.setdefault('output', {})['save_augmented_video'] = False
    if args.ball_backend is not None:
        overrides.setdefault('models', {}).setdefault('ball_detector', {})['backend'] = args.ball_backend
    if args.ball_weights is not None:
        overrides.setdefault('models', {}).setdefault('ball_detector', {})['weights'] = args.ball_weights
        overrides.setdefault('models', {}).setdefault('ball_detector', {})['backend'] = 'ultralytics'
    cfg = deep_update(cfg, overrides)
    run_analysis(cfg)


if __name__ == '__main__':
    main()
