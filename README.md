# AI Tennis Point Analyzer

Offline backend analyzer for tennis point videos. It produces an augmented video and structured CSV/JSON reports.

## Main outputs

For each input point video, the analyzer writes:

- `augmented_video.mp4` — video with ball trajectory, player skeleton, player movement trail, HIT/BOUNCE markers, and running stats overlay.
- `frame_report.csv` / `frame_report.json` — frame-level ball, player, pose, and event predictions.
- `events.csv` / `events.json` — event-level HIT/BOUNCE timeline.
- `point_summary.csv` / `point_summary.json` — per-point and per-player statistics.

## Quick start

```bash
conda create -n tennis_analyzer python=3.10 -y
conda activate tennis_analyzer
pip install -r requirements.txt

python -m src.main \
  --video data/raw_points/your_point_video.mp4 \
  --output_root outputs/runs \
  --point_id point_0001
```

The first run with Ultralytics will download model weights if they are not already present.

## Recommended model plan for 2 x RTX 4090

- Baseline implementation in this zip:
  - player detection: Ultralytics YOLOv8x person detector
  - pose: Ultralytics YOLOv8x-pose
  - ball: simple HSV detector or custom Ultralytics ball model if weights are provided
- Strong upgrade path:
  - player detection: YOLO11x or YOLOv8x
  - player tracking: ByteTrack
  - pose: RTMPose-l/x or ViTPose
  - ball: TrackNet-style heatmap tracker fine-tuned on tennis frames
  - court: tennis court keypoint detector + homography
  - events: temporal classifier over ball/pose/court sequences

## Current v0.1 limitations

This package is a clean backend scaffold with a runnable baseline. The ball detector defaults to a simple HSV threshold method, which is not production-grade. For real tennis videos, you should train or plug in a tennis-specific ball detector/tracker. HIT and BOUNCE detection are rule-based baselines.
