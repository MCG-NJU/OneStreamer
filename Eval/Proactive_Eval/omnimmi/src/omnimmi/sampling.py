from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: str
    source_fps: float
    total_frames: int
    duration: float


@dataclass(frozen=True)
class SamplingPlan:
    method: str
    start_seconds: float
    end_seconds: float
    requested_end_seconds: float
    sample_fps: float
    indices: tuple[int, ...]
    sample_times_seconds: tuple[float, ...]
    truncated: bool
    requested_points: int


def recent_frames_plan(
    *,
    end_seconds: float,
    source_fps: float,
    total_frames: int,
    target_fps: float,
    recent_frames: int,
) -> SamplingPlan:
    if source_fps <= 0 or target_fps <= 0 or total_frames <= 0:
        raise ValueError("source_fps, target_fps and total_frames must be positive")
    if recent_frames <= 0:
        raise ValueError("recent_frames must be positive")

    duration = total_frames / source_fps
    effective_end = min(max(float(end_seconds), 0.0), duration)
    clip_start = 0
    clip_end = min(total_frames, int(math.ceil(effective_end * source_fps)))
    clip_end = max(clip_start + 1, clip_end)
    clip_end = min(total_frames, clip_end)

    stride = source_fps / target_fps
    indices = []
    for offset in range(recent_frames):
        frame_index = clip_end - 1 - int(round(offset * stride))
        if frame_index < clip_start:
            break
        indices.append(frame_index)
    indices = sorted(set(indices))
    if not indices:
        indices = [clip_end - 1]
    actual_times = [index / source_fps for index in indices]

    return SamplingPlan(
        method="recent_frames",
        start_seconds=actual_times[0],
        end_seconds=effective_end,
        requested_end_seconds=float(end_seconds),
        sample_fps=float(target_fps),
        indices=tuple(indices),
        sample_times_seconds=tuple(actual_times),
        truncated=False,
        requested_points=recent_frames,
    )


def probe_video(path: str | os.PathLike[str]) -> VideoInfo:
    import decord
    from decord import cpu

    resolved = str(Path(path).resolve())
    reader = decord.VideoReader(resolved, ctx=cpu(0), num_threads=1)
    source_fps = float(reader.get_avg_fps())
    total_frames = len(reader)
    if source_fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Invalid video metadata for {resolved}: fps={source_fps}, frames={total_frames}"
        )
    return VideoInfo(
        path=resolved,
        source_fps=source_fps,
        total_frames=total_frames,
        duration=total_frames / source_fps,
    )


def _cache_key(video: VideoInfo, plan: SamplingPlan) -> str:
    payload = {
        "video": video.path,
        "source_fps": video.source_fps,
        "total_frames": video.total_frames,
        "plan": asdict(plan),
        "cache_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}"
    )


def _publish_cache_file(temporary: Path, path: Path) -> None:
    try:
        os.replace(temporary, path)
    except (FileExistsError, FileNotFoundError):
        if not path.is_file():
            raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_save_jpeg(image, path: Path) -> None:
    temporary = _temporary_path(path)
    image.save(temporary, format="JPEG", quality=95)
    _publish_cache_file(temporary, path)


def materialize_frames(
    video: VideoInfo,
    plan: SamplingPlan,
    cache_root: str | os.PathLike[str],
    decode_batch_size: int = 64,
) -> tuple[list[str], dict]:
    from PIL import Image
    import decord
    from decord import cpu

    cache_dir = Path(cache_root) / _cache_key(video, plan)
    cache_dir.mkdir(parents=True, exist_ok=True)
    count = len(plan.indices)
    frame_paths = [
        cache_dir / f"frame-{position:05d}-of-{count:05d}.jpg"
        for position in range(1, count + 1)
    ]

    missing_positions = [index for index, path in enumerate(frame_paths) if not path.is_file()]
    if missing_positions:
        reader = decord.VideoReader(video.path, ctx=cpu(0), num_threads=1)
        for offset in range(0, len(missing_positions), decode_batch_size):
            positions = missing_positions[offset : offset + decode_batch_size]
            source_indices = [plan.indices[position] for position in positions]
            arrays = reader.get_batch(source_indices).asnumpy()
            for position, array in zip(positions, arrays):
                if not frame_paths[position].is_file():
                    _atomic_save_jpeg(Image.fromarray(array), frame_paths[position])

    metadata = {
        "video": asdict(video),
        "sampling": {
            **asdict(plan),
            "indices": list(plan.indices),
            "sample_times_seconds": list(plan.sample_times_seconds),
            "sample_n_frames": count,
        },
    }
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        temporary = _temporary_path(metadata_path)
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _publish_cache_file(temporary, metadata_path)
    return [str(path) for path in frame_paths], metadata
