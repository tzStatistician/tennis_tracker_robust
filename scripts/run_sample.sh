#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH=${1:-data/raw_points/sample.mp4}
POINT_ID=${2:-sample_point}

python -m src.main \
  --video "$VIDEO_PATH" \
  --output_root outputs/runs \
  --point_id "$POINT_ID"
