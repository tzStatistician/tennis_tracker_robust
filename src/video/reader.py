from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple
import cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    total_frames: int


class VideoReader:
    def __init__(self, video_path: str | Path):
        self.path = str(video_path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Cannot open video: {self.path}')
        self.info = VideoInfo(
            path=self.path,
            fps=float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self, every_n: int = 1, max_frames: int | None = None) -> Iterator[Tuple[int, float, object]]:
        n = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_idx % every_n != 0:
                continue
            ts = frame_idx / max(self.info.fps, 1e-9)
            yield frame_idx, ts, frame
            n += 1
            if max_frames is not None and n >= max_frames:
                break

    def release(self) -> None:
        self.cap.release()
