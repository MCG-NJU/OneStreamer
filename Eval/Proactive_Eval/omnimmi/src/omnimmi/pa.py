from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from .common import sample_id, config_fingerprint, source_video_path
def _round_span(round_index: int, fps: float) -> list[float]:
    return [round_index / fps, (round_index + 1) / fps]

def _parse_active_response(raw: str) -> tuple[bool, str | None]:
    value = (raw or "").strip()
    tag = "</Response>"
    if not value.startswith(tag):
        return False, None
    return True, value[len(tag) :].lstrip(" \t:\n")

def _infer_pa_sample(
    model,
    config: dict[str, Any],
    index: int,
    sample: dict[str, Any],
) -> dict[str, Any]:
    fps = float(config.get("target_fps", 1.0))
    video_path = source_video_path(config["data_root"], sample["video"])
    model.reset_session(question=sample["question"], extra_turns=[])
    extractor = model._open_extractor(str(video_path))
    records = []
    response_times = []
    try:
        total_rounds = extractor.get_total_rounds()
        for round_index in range(total_rounds):
            try:
                frame = extractor.get_frame_at_round(round_index)
            except StopIteration:
                break
            raw = str(model.step(frame, round_idx=round_index))
            is_response, content = _parse_active_response(raw)
            span = _round_span(round_index, fps)
            records.append(
                {
                    "round_index": round_index,
                    "video_span": span,
                    "answerable": "Yes" if is_response else "No",
                    "model_response": content,
                    "raw_answer": raw,
                }
            )
            if is_response and content:
                response_times.append(span[1])
    finally:
        extractor.close()

    result = copy.deepcopy(sample)
    result.update(
        {
            "sample_id": sample_id("pa", index),
            "task": "pa",
            "suite": config["suite"],
            "protocol": "proactive",
            "target_fps": fps,
            "max_context_rounds": int(config.get("max_rounds", 32)),
            "config_fingerprint": config_fingerprint(config),
            "response_times_seconds": response_times,
            "pred": response_times,
            "records": records,
        }
    )
    return result
