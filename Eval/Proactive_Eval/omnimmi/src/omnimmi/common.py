from __future__ import annotations

import json

import hashlib

import os

import random

from pathlib import Path

from typing import Any, Iterable

ANNOTATION_FILES = {
    "ap": "action_prediction.json",
    "si": "speaker_identification.json",
    "md": "multiturn_dependency_reasoning.json",
    "sg": "dynamic_state_grounding.json",
    "pa": "proactive_alerting.json",
}

MAIN_TASKS = ("ap", "si", "md", "sg")

def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def load_annotations(data_root: str | os.PathLike[str], task: str) -> list[dict[str, Any]]:
    path = annotation_path(data_root, task)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {path}, got {type(data).__name__}")
    return data

def source_video_path(data_root: str | os.PathLike[str], video_name: str) -> Path:
    path = Path(data_root) / "videos" / video_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing OmniMMI video: {path}")
    return path

def parse_span(value: str) -> tuple[float, float]:
    parts = value.split("--")
    if len(parts) != 2:
        raise ValueError(f"Invalid OmniMMI timestamp span: {value!r}")
    start, end = (float(part.strip()) for part in parts)
    return start, end

def sample_id(task: str, index: int) -> str:
    return f"{task}:{index:05d}"

def annotation_path(data_root: str | os.PathLike[str], task: str) -> Path:
    if task not in ANNOTATION_FILES:
        raise ValueError(f"Unsupported OmniMMI task: {task}")
    root = Path(data_root)
    filename = ANNOTATION_FILES[task]
    candidates = (root / filename, root / "annotations" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {task} annotations; tried: {tried}")
