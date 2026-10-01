from __future__ import annotations

from dataclasses import dataclass

from typing import Any

CAPTION_PROMPT_VERSION = "hierarchical_caption_global_instruction_v2"

CAPTION_INSTRUCTION_GLOBAL = True

MEMORY_QA_PROMPT_VERSION = "post_observation_memory_qa_mixed_v3"







V50_TRAINING_CAPTION_SYSTEM = """You are a streaming visual captioner. Process each video update using only visible evidence.
Use </Silence> alone when no visual unit is ready to report.
Use </Observe> for completed visual details, with timestamps in seconds to one decimal place: [16.0 seconds] for a moment or [0.0 - 20.0 seconds] for an interval.
Use </Summary> with [0.0 - 20.0 seconds] for a completed semantic segment, after all its details and before any details of the next segment.
Emit one kind of action per turn. Multiple details in one action retain their individual prefixes and timestamps.
At <STREAM_END>, finish pending active captions in separate turns without advancing video time.
Only after an explicit user question about the complete video, use </Response> Overall Summary: followed by the overall account.
Never predict unseen content."""

@dataclass(frozen=True)
class CaptionControlProtocol:
    name: str
    system: str
    instruction: str
    tag_to_kind: dict[str, str]

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(f"</{tag}>" for tag in self.tag_to_kind)


def get_caption_control_protocol(
    caption_config: dict[str, Any] | None = None,
    control_protocol: str | None = None,
) -> CaptionControlProtocol:
    name = control_protocol or (caption_config or {}).get(
        "control_protocol", DEFAULT_CONTROL_PROTOCOL
    )
    try:
        return CAPTION_CONTROL_PROTOCOLS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown caption control protocol {name!r}; "
            f"choices: {sorted(CAPTION_CONTROL_PROTOCOLS)}"
        ) from exc

def validate_caption_control_tokens(tokenizer, protocol: CaptionControlProtocol) -> None:

    failures = []
    for token in protocol.tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        if not token_ids or decoded != token:
            failures.append(f"{token}: ids={token_ids!r}, decoded={decoded!r}")
    if failures:
        raise RuntimeError(
            f"Caption protocol {protocol.name!r} special-token preflight failed: "
            + "; ".join(failures)
        )

OVERALL_SUMMARY_INSTRUCTION = (
    "The observed portion of the video has ended. Based only on the timestamped "
    "observations above, provide a concise overall summary of what happened. Preserve "
    "important entities, actions, state changes, temporal order, and details that may "
    "help answer a later question. Do not infer unseen content. Follow the system "
    "protocol for an explicit overall-summary request."
)


MEMORY_QA_SYSTEM = """You answer a multiple-choice question about a video after the observation period has ended.
You may receive two evidence sources:
1. A causal textual memory produced while the video was playing, without access to the current question. It may be incomplete or imperfect; use it to recover earlier events.
2. Recent video frames from immediately before the question time, ordered from oldest to newest. Use them for the latest visual state.

Use the recent frames as stronger evidence when they conflict with the textual memory. Base the answer only on the supplied memory and frames, select the single best option, and output only its option letter."""

MEMORY_DIRECT_QA_SYSTEM = """You answer a question about a video after the observation period has ended.
You may receive two evidence sources:
1. A causal textual memory produced while the video was playing, without access to the current question. It may be incomplete or imperfect; use it to recover earlier events.
2. Recent video frames from immediately before the question time, ordered from oldest to newest. Use them for the latest visual state.

Use the recent frames as stronger evidence when they conflict with the textual memory. Base the answer only on the supplied memory and frames. Follow the response format requested by the question exactly and output only the requested answer."""

MEMORY_PREAMBLE = """<LONG_TERM_VIDEO_MEMORY>
This memory contains only observations generated before the question time.
Times are relative to the beginning of the evaluated video clip."""

MEMORY_POSTAMBLE = "</LONG_TERM_VIDEO_MEMORY>"


DEFAULT_CONTROL_PROTOCOL = "silence_observe_summary_v50_training_user"
CAPTION_CONTROL_PROTOCOLS = {DEFAULT_CONTROL_PROTOCOL: CaptionControlProtocol(
    name=DEFAULT_CONTROL_PROTOCOL, system=V50_TRAINING_CAPTION_SYSTEM,
    instruction="<VIDEO_CONTEXT>\nContinuously observe the video, use </Observe> for completed visual details, and use </Summary> to summarize each completed semantic segment.",
    tag_to_kind={"Silence":"standby","Observe":"response","Summary":"summary"})}
