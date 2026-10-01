from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import parse_span, source_video_path
from .sampling import (
    SamplingPlan,
    VideoInfo,
    recent_frames_plan,
)


@dataclass(frozen=True)
class VisualRequest:
    media_path: Path
    semantic_target_seconds: float
    target_clamped: bool
    plan: SamplingPlan


def ap_prompt(sample: dict[str, Any]) -> str:
    return (
        f"{sample['question']}, Please select the next step from the provided "
        f"vocabulary list: {sample['vocab']}."
    )


def md_prompt(question: str, previous_prediction: str | None) -> str:
    if previous_prediction is None:
        return question
    return question.replace("##ANSWER##", previous_prediction)


def sg_prompt(history: list[tuple[str, str]], question: str) -> str:
    parts: list[str] = []
    for old_question, old_answer in history:
        parts.extend(("user", old_question, "assistant", old_answer))
    parts.extend(("user", question, "assistant"))
    return "\n".join(parts) + "\n"


def semantic_target_seconds(
    protocol: str,
    task: str,
    sample: dict[str, Any],
    video_duration: float,
    qa: dict[str, Any] | None,
    ap_lead_seconds: float = 0.0,
) -> float:
    if protocol != "recent_window":
        raise ValueError(f"Unknown protocol: {protocol}")
    if task == "ap":
        return max(0.0, video_duration - ap_lead_seconds)
    if task == "si":
        return float(sample["timestamp"])
    if task in ("md", "sg") and qa is not None:
        start, end = parse_span(qa["timestamp"])
        return (start + end) / 2.0
    raise ValueError(f"Cannot resolve target for task={task}, qa={qa is not None}")


def build_visual_request(
    *,
    config: dict[str, Any],
    task: str,
    sample: dict[str, Any],
    source_video: VideoInfo,
    media_video: VideoInfo,
    target_fps: float,
    recent_frames: int | None = None,
    qa: dict[str, Any] | None = None,
) -> VisualRequest:
    protocol = config["protocol"]
    semantic_target = semantic_target_seconds(
        protocol,
        task,
        sample,
        source_video.duration,
        qa,
        float(config.get("ap_lead_seconds", 0.0)),
    )


    if protocol == "recent_window":
        if recent_frames is None:
            raise ValueError("recent_window requires an explicit recent_frames variant")
        clamped_target = min(max(semantic_target, 0.0), source_video.duration)
        plan = recent_frames_plan(
            end_seconds=clamped_target,
            source_fps=media_video.source_fps,
            total_frames=media_video.total_frames,
            target_fps=target_fps,
            recent_frames=recent_frames,
        )
        return VisualRequest(
            media_path=Path(media_video.path),
            semantic_target_seconds=semantic_target,
            target_clamped=not abs(clamped_target - semantic_target) < 1e-9,
            plan=plan,
        )
    raise ValueError(f"Unknown protocol: {protocol}")


def resolve_media_path(
    config: dict[str, Any], task: str, sample: dict[str, Any], qa: dict[str, Any] | None
) -> Path:
    data_root = config["data_root"]
    return source_video_path(data_root, sample["video"])
