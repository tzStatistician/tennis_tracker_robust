from __future__ import annotations

from pathlib import Path
import cv2


class VideoWriter:
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        self.output_path = str(output_path)
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f'Cannot create video writer: {self.output_path}')

    def write(self, frame) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
