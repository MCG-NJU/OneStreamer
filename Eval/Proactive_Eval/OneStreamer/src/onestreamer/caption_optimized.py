from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .caption import TimestampedCaptionSession, boundary_key
from .memory import MemoryBank, MemoryEvent, parse_caption_output
from .prompts import (
    CAPTION_INSTRUCTION_GLOBAL,
    get_caption_control_protocol,
    validate_caption_control_tokens,
)
from .schema import Sample
from .video import iter_stream_frames


CAPTION_PROTOCOL_VERSION = "hierarchical_boundary_opt_v1"


@dataclass
class SegmentScopeTracker:

    fallback_start: float
    control_protocol: str | None = None
    first_response_start: float | None = None

    @property
    def segment_start(self) -> float:
        if self.first_response_start is not None:
            return self.first_response_start
        return self.fallback_start

    def parse_online(self, raw: str, turn_start: float, turn_end: float) -> MemoryEvent:
        parsed = parse_caption_output(
            raw, turn_start, turn_end, control_protocol=self.control_protocol
        )
        if parsed.kind == "response" and parsed.text:
            if self.first_response_start is None:
                self.first_response_start = turn_start
            return parsed
        if parsed.kind != "summary" or not parsed.text:
            return parsed
        event = MemoryEvent(
            kind="summary",
            start=min(self.segment_start, turn_end),
            end=turn_end,
            text=parsed.text,
            raw=parsed.raw,
        )
        self.fallback_start = turn_end
        self.first_response_start = None
        return event




def infer_open_segment_start(events: list[MemoryEvent], stream_start: float) -> float:
    fallback_start = stream_start
    for event in events:
        if event.kind == "summary":
            fallback_start = event.end
    for event in events:
        if event.kind == "response" and event.start >= fallback_start - 1e-6:
            return event.start
    return fallback_start


def _caption_session(
    engine,
    sample: Sample,
    caption_config: dict[str, Any],
) -> TimestampedCaptionSession:
    from .caption_video import VideoCaptionSession
    return VideoCaptionSession(engine, sample, caption_config)


def _caption_updates(video_path, sample, config):
    from .caption_video import iter_video_updates
    return iter_video_updates(video_path, sample.start, sample.end, float(config['fps']))


def _output_record(
    round_value: int | str,
    turn_start: float,
    turn_end: float,
    event: MemoryEvent,
    raw: str,
) -> dict[str, Any]:
    return {
        "round": round_value,
        "start": turn_start,
        "end": turn_end,
        "kind": event.kind,
        "memory_start": event.start,
        "memory_end": event.end,
        "output": raw,
    }


def _stream_end_result(
    raw: str,
    segment_start: float,
    end: float,
    control_protocol: str | None = None,
) -> tuple[MemoryEvent | None, dict[str, Any]]:
    parsed = parse_caption_output(
        raw, segment_start, end, control_protocol=control_protocol
    )
    accepted = parsed.kind == "summary" and bool(parsed.text)
    event = parsed if accepted else None
    return event, {
        "round": "stream_end",
        "start": segment_start,
        "end": end,
        "kind": parsed.kind,
        "memory_start": segment_start,
        "memory_end": end,
        "scope_basis": "first_response_since_previous_summary_with_summary_boundary_fallback",
        "accepted": accepted,
        "output": raw,
    }


def finalize_caption_boundary(
    engine,
    session: TimestampedCaptionSession | None,
    causal_events: list[MemoryEvent],
    causal_outputs: list[dict[str, Any]],
    segment_start: float,
    start: float,
    end: float,
    caption_config: dict[str, Any],
    cached_stream_end_raw: str | None = None,
) -> dict[str, Any]:

    del start
    events = list(causal_events)
    boundary_outputs: list[dict[str, Any]] = []
    last_kind = causal_outputs[-1]["kind"] if causal_outputs else None
    should_flush = bool(caption_config.get("flush_summary", True)) and last_kind == "response"

    if should_flush:
        raw = cached_stream_end_raw
        if raw is None and hasattr(session, 'finish_boundary'):
            raw = session.finish_boundary()
        if raw is None:
            if session is None or session._data is None:
                raise RuntimeError("STREAM_END is required but no live session or cached output exists")
            data = {
                "images": list(session._data["images"]),
                "messages": list(session._data["messages"]),
            }
            if session._last_answer is not None:
                data["messages"].append({"role": "assistant", "content": session._last_answer})
            data["messages"].append({"role": "user", "content": "<STREAM_END>"})
            raw = engine.infer(
                data,
                max_tokens=int(caption_config.get("max_new_tokens", 128)),
                temperature=0.0,
            )
        event, record = _stream_end_result(
            raw,
            segment_start,
            end,
            control_protocol=caption_config.get("control_protocol"),
        )
        boundary_outputs.append(record)
        if event is not None:
            events.append(event)

    bank = MemoryBank(events)
    return {
        "question_time": end,
        "caption_protocol_version": CAPTION_PROTOCOL_VERSION,
        "segment_scope_start": segment_start,
        "counts": {
            "rounds": len(causal_outputs),
            "captions": len(bank.captions),
            "summaries": len(bank.summaries),
            "invalid": sum(item["kind"] == "invalid" for item in causal_outputs),
        },
        "events": [event.to_dict() for event in events],
        "boundary_outputs": boundary_outputs,
    }




def run_caption_pass(
    engine,
    sample: Sample,
    video_path: str | Path,
    caption_config: dict[str, Any],
) -> dict[str, Any]:
    fps = float(caption_config.get("fps", 1.0))
    if fps <= 0:
        raise ValueError("caption.fps must be positive")
    session = _caption_session(engine, sample, caption_config)
    tracker = SegmentScopeTracker(
        sample.start, control_protocol=caption_config.get("control_protocol")
    )
    outputs: list[dict[str, Any]] = []
    events: list[MemoryEvent] = []
    started = time.monotonic()
    for round_number, (start, end, frame) in enumerate(
        _caption_updates(video_path, sample, caption_config)
    ):
        raw = session.step(frame, round_idx=round_number)
        event = tracker.parse_online(raw, start, end)
        outputs.append(_output_record(round_number, start, end, event, raw))
        if event.kind in {"response", "summary"} and event.text:
            events.append(event)
        progress_every = int(caption_config.get("progress_every", 20))
        if event.kind in {"response", "summary", "invalid"} or (
            progress_every > 0 and (round_number + 1) % progress_every == 0
        ):
            print(
                f"caption_opt sample={sample.sample_id} round={round_number + 1} "
                f"time={end:.2f}/{sample.end:.2f}s kind={event.kind}",
                flush=True,
            )

    causal_events = list(events)
    causal_outputs = list(outputs)
    boundary = finalize_caption_boundary(
        engine=engine,
        session=session,
        causal_events=causal_events,
        causal_outputs=causal_outputs,
        segment_start=tracker.segment_start,
        start=sample.start,
        end=sample.end,
        caption_config=caption_config,
    )
    outputs.extend(boundary["boundary_outputs"])
    return {
        "sample_id": sample.sample_id,
        "video": str(video_path),
        "question_time": sample.end,
        "caption_fps": fps,
        "caption_context_frames": int(caption_config.get("context_frames", 32)),
        "caption_instruction_global": CAPTION_INSTRUCTION_GLOBAL,
        "caption_control_protocol": get_caption_control_protocol(caption_config).name,
        "caption_protocol_version": CAPTION_PROTOCOL_VERSION,
        "runtime_seconds": time.monotonic() - started,
        "counts": boundary["counts"],
        "events": boundary["events"],
        "causal_events": [event.to_dict() for event in causal_events],
        "outputs": outputs,
    }


def run_shared_caption_pass(
    engine,
    sample: Sample,
    video_path: str | Path,
    boundaries: list[float],
    caption_config: dict[str, Any],
) -> dict[str, Any]:

    fps = float(caption_config.get("fps", 1.0))
    if fps <= 0:
        raise ValueError("caption.fps must be positive")
    boundaries = sorted(set(float(value) for value in boundaries))
    if not boundaries or boundaries[-1] != sample.end:
        raise ValueError("Shared caption sample.end must equal the largest boundary")
    for boundary in boundaries:
        round_fps = 1.0 if caption_config.get('input_mode') == 'video_per_second' else fps
        grid_position = (boundary - sample.start) * round_fps
        if boundary <= sample.start or abs(grid_position - round(grid_position)) > 1e-6:
            raise ValueError(
                f"Question boundary {boundary} is not aligned to the {fps:g} fps caption grid"
            )

    session = _caption_session(engine, sample, caption_config)
    tracker = SegmentScopeTracker(
        sample.start, control_protocol=caption_config.get("control_protocol")
    )
    outputs: list[dict[str, Any]] = []
    events: list[MemoryEvent] = []
    boundary_results: dict[str, dict[str, Any]] = {}
    pending = iter(boundaries)
    next_boundary = next(pending, None)
    started = time.monotonic()
    for round_number, (start, end, frame) in enumerate(
        _caption_updates(video_path, sample, caption_config)
    ):
        raw = session.step(frame, round_idx=round_number)
        event = tracker.parse_online(raw, start, end)
        outputs.append(_output_record(round_number, start, end, event, raw))
        if event.kind in {"response", "summary"} and event.text:
            events.append(event)
        progress_every = int(caption_config.get("progress_every", 20))
        if event.kind in {"response", "summary", "invalid"} or (
            progress_every > 0 and (round_number + 1) % progress_every == 0
        ):
            print(
                f"caption_opt sample={sample.sample_id} round={round_number + 1} "
                f"time={end:.2f}/{sample.end:.2f}s kind={event.kind}",
                flush=True,
            )

        while next_boundary is not None and end >= next_boundary - 1e-6:
            causal_outputs = [item for item in outputs if item["end"] <= next_boundary + 1e-6]
            causal_events = [item for item in events if item.end <= next_boundary + 1e-6]
            segment_start = infer_open_segment_start(causal_events, sample.start)
            boundary_results[boundary_key(next_boundary)] = finalize_caption_boundary(
                engine=engine,
                session=session,
                causal_events=causal_events,
                causal_outputs=causal_outputs,
                segment_start=segment_start,
                start=sample.start,
                end=next_boundary,
                caption_config=caption_config,
            )
            next_boundary = next(pending, None)

    if next_boundary is not None:
        raise RuntimeError(f"Video ended before question boundary {next_boundary}")
    return {
        "sample_id": sample.sample_id,
        "video": str(video_path),
        "stream_start": sample.start,
        "stream_end": sample.end,
        "boundaries": boundaries,
        "caption_fps": fps,
        "caption_context_frames": int(caption_config.get("context_frames", 32)),
        "caption_instruction_global": CAPTION_INSTRUCTION_GLOBAL,
        "caption_control_protocol": get_caption_control_protocol(caption_config).name,
        "caption_protocol_version": CAPTION_PROTOCOL_VERSION,
        "runtime_seconds": time.monotonic() - started,
        "causal_events": [event.to_dict() for event in events],
        "outputs": outputs,
        "boundary_results": boundary_results,
    }
