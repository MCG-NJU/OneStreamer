from __future__ import annotations

import json

import os

import socket

import time

import traceback

from dataclasses import replace

from pathlib import Path

from typing import Any

from .caption import boundary_key
from .caption_optimized import run_caption_pass, run_shared_caption_pass

from .engine import RECENT_FRAME_INPUT_VERSION, OneStreamerEngine


from .io import append_jsonl, atomic_write_json, read_jsonl, stable_fingerprint

from .memory import MemoryBank, MemoryEvent

from .ovobench import build_question_text, score_prediction

from .prompts import (
    CAPTION_INSTRUCTION_GLOBAL,
    CAPTION_PROMPT_VERSION,
    MEMORY_DIRECT_QA_SYSTEM,
    MEMORY_QA_SYSTEM,
    MEMORY_QA_PROMPT_VERSION,
    OVERALL_SUMMARY_INSTRUCTION,
    get_caption_control_protocol,
)

from .schema import ExperimentConfig, Sample, load_samples

from .video import probe_video, read_recent_frames

def caption_fingerprint(
    config: ExperimentConfig,
    model_alias: str,
    sample: Sample,
    video_path: Path,
) -> str:
    stat = video_path.stat()
    protocol = get_caption_control_protocol(config.caption)
    return stable_fingerprint(
        {
            "schema": 3,
            "model_alias": model_alias,
            "model_path": config.model_path(model_alias),
            "sample_id": sample.sample_id,
            "start": sample.start,
            "end": sample.end,
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "caption": config.caption,
            "caption_prompt_version": CAPTION_PROMPT_VERSION,
            "caption_instruction_global": CAPTION_INSTRUCTION_GLOBAL,
            "caption_control_protocol": protocol.name,
            "caption_system": protocol.system,
            "caption_instruction": protocol.instruction,
            "overall_summary_instruction": OVERALL_SUMMARY_INSTRUCTION,
        }
    )

def caption_cache_path(config: ExperimentConfig, model_alias: str, sample: Sample) -> Path:
    return config.cache_root / "captions" / model_alias / f"{sample.sample_id}.json"

def load_or_create_caption(
    engine: OneStreamerEngine,
    config: ExperimentConfig,
    model_alias: str,
    sample: Sample,
    video_path: Path,
    allow_create: bool,
) -> tuple[dict[str, Any], bool, Path]:
    path = caption_cache_path(config, model_alias, sample)
    fingerprint = caption_fingerprint(config, model_alias, sample, video_path)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return cached, True, path
    if not allow_create:
        raise FileNotFoundError(f"Missing valid caption cache: {path}")
    result = run_caption_pass(engine, sample, video_path, config.caption)
    result.update(
        {
            "fingerprint": fingerprint,
            "model_alias": model_alias,
            "model_path": config.model_path(model_alias),
        }
    )
    atomic_write_json(path, result)
    return result, False, path

def _stream_key(sample: Sample) -> tuple[str, float]:
    return sample.video, sample.start

def shared_caption_cache_path(
    config: ExperimentConfig,
    model_alias: str,
    samples: list[Sample],
) -> Path:
    identity = stable_fingerprint(
        {
            "video": samples[0].video,
            "start": samples[0].start,
            "boundaries": sorted({sample.end for sample in samples}),
        }
    )
    return config.cache_root / "shared_timelines" / model_alias / f"stream-{identity[:24]}.json"

def shared_caption_fingerprint(
    config: ExperimentConfig,
    model_alias: str,
    samples: list[Sample],
    video_path: Path,
) -> str:
    stat = video_path.stat()
    protocol = get_caption_control_protocol(config.caption)
    return stable_fingerprint(
        {
            "schema": 1,
            "model_alias": model_alias,
            "model_path": config.model_path(model_alias),
            "video": samples[0].video,
            "start": samples[0].start,
            "boundaries": sorted({sample.end for sample in samples}),
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "caption": config.caption,
            "caption_prompt_version": CAPTION_PROMPT_VERSION,
            "caption_instruction_global": CAPTION_INSTRUCTION_GLOBAL,
            "caption_control_protocol": protocol.name,
            "caption_system": protocol.system,
            "caption_instruction": protocol.instruction,
            "overall_summary_instruction": OVERALL_SUMMARY_INSTRUCTION,
        }
    )

def load_or_create_shared_caption(
    engine: OneStreamerEngine,
    config: ExperimentConfig,
    model_alias: str,
    sample: Sample,
    stream_samples: list[Sample],
    video_path: Path,
    allow_create: bool,
    timeline_memory: dict[Path, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool, Path]:
    if any(_stream_key(item) != _stream_key(sample) for item in stream_samples):
        raise ValueError(f"Mixed streams supplied for {sample.sample_id}")
    path = shared_caption_cache_path(config, model_alias, stream_samples)
    fingerprint = shared_caption_fingerprint(config, model_alias, stream_samples, video_path)
    reused = False
    timeline = timeline_memory.get(path) if timeline_memory is not None else None
    if timeline is not None and timeline.get("fingerprint") == fingerprint:
        reused = True
    else:
        timeline = None
    if timeline is None and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            timeline = cached
            reused = True
    if timeline is None:
        if not allow_create:
            raise FileNotFoundError(f"Missing valid shared caption timeline: {path}")
        max_end = max(item.end for item in stream_samples)
        template = replace(
            max(stream_samples, key=lambda item: item.end),
            sample_id=f"stream-{path.stem.removeprefix('stream-')}",
            end=max_end,
        )
        timeline = run_shared_caption_pass(
            engine=engine,
            sample=template,
            video_path=video_path,
            boundaries=sorted({item.end for item in stream_samples}),
            caption_config=config.caption,
        )
        timeline.update(
            {
                "fingerprint": fingerprint,
                "model_alias": model_alias,
                "model_path": config.model_path(model_alias),
            }
        )
        atomic_write_json(path, timeline)
    if timeline_memory is not None:
        timeline_memory[path] = timeline

    key = boundary_key(sample.end)
    if key not in timeline["boundary_results"]:
        raise KeyError(f"Shared timeline {path} has no boundary {key} for {sample.sample_id}")
    boundary = timeline["boundary_results"][key]
    result = {
        "sample_id": sample.sample_id,
        "video": str(video_path),
        "question_time": sample.end,
        "caption_fps": timeline["caption_fps"],
        "caption_context_frames": timeline["caption_context_frames"],
        "caption_instruction_global": timeline["caption_instruction_global"],
        "caption_control_protocol": timeline.get(
            "caption_control_protocol", get_caption_control_protocol(config.caption).name
        ),
        "counts": boundary["counts"],
        "events": boundary["events"],
        "boundary_outputs": boundary["boundary_outputs"],
        "shared_timeline_cache": str(path),
    }
    return result, reused, path
