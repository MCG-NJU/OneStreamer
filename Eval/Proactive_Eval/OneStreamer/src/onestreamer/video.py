from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


def probe_video(path: str | Path) -> VideoInfo:
    import cv2

    path = str(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or total <= 0:
        raise ValueError(f"Invalid video metadata: path={path} fps={fps} frames={total}")
    return VideoInfo(path, fps, total, total / fps, width, height)


def _to_pil(frame):
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def iter_stream_frames(
    path: str | Path,
    start_seconds: float,
    end_seconds: float,
    target_fps: float,
) -> Iterator[tuple[float, float, object]]:
    import cv2

    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    info = probe_video(path)
    clip_start = max(0, int(math.floor(start_seconds * info.fps)))
    clip_end = min(info.total_frames, int(math.ceil(end_seconds * info.fps)))
    count = max(1, int(math.ceil((end_seconds - start_seconds) * target_fps)))
    indices = []
    for offset in range(count):
        timestamp = start_seconds + offset / target_fps
        if timestamp >= end_seconds:
            break
        index = min(clip_end - 1, max(clip_start, int(math.floor(timestamp * info.fps))))
        if not indices or index != indices[-1][0]:
            indices.append((index, timestamp))

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        current_index = clip_start
        for index, timestamp in indices:
            frame = None
            while current_index <= index:
                ok, decoded = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {current_index} from {path}")
                if current_index == index:
                    frame = decoded
                current_index += 1
            if frame is None:
                raise RuntimeError(f"Internal sampling error at frame {index} from {path}")
            interval_end = min(end_seconds, timestamp + 1.0 / target_fps)
            yield timestamp, interval_end, _to_pil(frame)
    finally:
        capture.release()


def read_recent_frames(
    path: str | Path,
    start_seconds: float,
    end_seconds: float,
    recent_frames: int,
    sample_fps: float = 1.0,
) -> list[tuple[float, object]]:
    import cv2

    if recent_frames <= 0 or sample_fps <= 0:
        raise ValueError("recent_frames and sample_fps must be positive")
    info = probe_video(path)
    clip_start = max(0, int(math.floor(start_seconds * info.fps)))
    clip_end = min(info.total_frames, int(math.ceil(end_seconds * info.fps)))
    stride = info.fps / sample_fps
    indices = []
    for offset in range(recent_frames):
        index = clip_end - 1 - int(round(offset * stride))
        if index < clip_start:
            break
        indices.append(index)
    indices = sorted(set(indices)) or [clip_end - 1]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    result = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode frame {index} from {path}")
            result.append((index / info.fps, _to_pil(frame)))
    finally:
        capture.release()
    return result
