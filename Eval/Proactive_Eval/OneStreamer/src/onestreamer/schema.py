from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    sample_id: str
    index: int
    task: str
    subtask: str
    video: str
    start: float
    end: float
    question: str
    options: dict[str, str]
    answer: str
    answer_text: str
    eval_mode: str = "mcq_letter"
    prepared_shard: int | None = None
    prepared_num_shards: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sample":
        options = {
            str(k).upper(): str(v)
            for k, v in value.get("options", {}).items()
            if str(v).strip()
        }
        eval_mode = str(value.get("eval_mode", "mcq_letter")).strip()
        if eval_mode not in {"mcq_letter", "substring_match", "integer_match"}:
            raise ValueError(f"{value['sample_id']}: unsupported eval_mode {eval_mode!r}")
        answer = str(value["answer"]).strip()
        if eval_mode == "mcq_letter":
            answer = answer.upper()
            if answer not in options:
                raise ValueError(f"{value['sample_id']}: answer is not present in options")
        start, end = float(value.get("start", 0.0)), float(value["end"])
        if end <= start:
            raise ValueError(f"{value['sample_id']}: end must be after start")
        prepared_shard = value.get("prepared_shard")
        prepared_num_shards = value.get("prepared_num_shards")
        if (prepared_shard is None) != (prepared_num_shards is None):
            raise ValueError(
                f"{value['sample_id']}: prepared_shard and prepared_num_shards must be set together"
            )
        if prepared_shard is not None:
            prepared_shard, prepared_num_shards = int(prepared_shard), int(prepared_num_shards)
            if not 0 <= prepared_shard < prepared_num_shards:
                raise ValueError(f"{value['sample_id']}: invalid prepared shard metadata")
        return cls(
            sample_id=str(value["sample_id"]),
            index=int(value["index"]),
            task=str(value["task"]),
            subtask=str(value.get("subtask", "")),
            video=str(value["video"]),
            start=start,
            end=end,
            question=str(value["question"]),
            options=options,
            answer=answer,
            answer_text=str(value.get("answer_text", options.get(answer, answer))),
            eval_mode=eval_mode,
            prepared_shard=prepared_shard,
            prepared_num_shards=prepared_num_shards,
        )


@dataclass(frozen=True)
class Variant:
    name: str
    memory_mode: str
    recent_frames: int
    recent_fps: float | None = None
    recent_resize_policy: str = "source_pixels"


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    models: dict[str, str]
    video_root: Path
    samples_path: Path
    cache_root: Path
    caption: dict[str, Any]
    answer: dict[str, Any]
    variants: tuple[Variant, ...]

    def model_path(self, alias: str) -> str:
        if alias not in self.models:
            raise KeyError(f"Unknown model alias {alias!r}; choices: {sorted(self.models)}")
        return self.models[alias]

    def recent_fps(self, variant: Variant) -> float:
        if variant.recent_fps is not None:
            return variant.recent_fps
        return float(self.answer.get("recent_sample_fps", 1.0))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    variants = tuple(
        Variant(
            name=str(item["name"]),
            memory_mode=str(item["memory_mode"]),
            recent_frames=int(item["recent_frames"]),
            recent_resize_policy=str(item.get("recent_resize_policy", "source_pixels")),
            recent_fps=(
                None if item.get("recent_fps") is None else float(item["recent_fps"])
            ),
        )
        for item in value["variants"]
    )
    names = [item.name for item in variants]
    if len(names) != len(set(names)):
        raise ValueError("Variant names must be unique")
    for item in variants:
        if item.recent_resize_policy not in {"source_pixels"}:
            raise ValueError(f"Unknown recent_resize_policy: {item.recent_resize_policy}")
        if item.recent_frames <= 0:
            raise ValueError(f"{item.name}: recent_frames must be positive")
        if item.recent_fps is not None and item.recent_fps <= 0:
            raise ValueError(f"{item.name}: recent_fps must be positive")
    config = ExperimentConfig(
        path=path,
        models={str(k): str(v) for k, v in value["models"].items()},
        video_root=_resolve(base, value["video_root"]),
        samples_path=_resolve(base, value["samples"]),
        cache_root=_resolve(base, value["cache_root"]),
        caption=dict(value["caption"]),
        answer=dict(value["answer"]),
        variants=variants,
    )
    for item in variants:
        if config.recent_fps(item) <= 0:
            raise ValueError(f"{item.name}: resolved recent fps must be positive")
    return config


def load_samples(path: str | Path) -> list[Sample]:
    samples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(Sample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid sample at {path}:{line_number}: {exc}") from exc
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return samples
