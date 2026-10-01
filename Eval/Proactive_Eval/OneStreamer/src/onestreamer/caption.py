from __future__ import annotations

import time

from pathlib import Path

from typing import Any

from shared.inference_fast import StreamingSession

from .memory import MemoryBank, MemoryEvent, parse_caption_output

from .prompts import (
    CAPTION_INSTRUCTION_GLOBAL,
    OVERALL_SUMMARY_INSTRUCTION,
    get_caption_control_protocol,
    validate_caption_control_tokens,
)

from .schema import Sample

from .video import iter_stream_frames

class TimestampedCaptionSession(StreamingSession):
    def __init__(self, *args, stream_start: float, stream_end: float, stream_fps: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_start = stream_start
        self.stream_end = stream_end
        self.stream_fps = stream_fps

    @staticmethod
    def _format_time(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:.3f}".rstrip("0").rstrip(".")

    def _user_content(self, round_idx: int, include_question: bool) -> str:
        start = self.stream_start + round_idx / self.stream_fps
        end = min(self.stream_end, start + 1.0 / self.stream_fps)
        time_tag = f"<{self._format_time(start)}s-{self._format_time(end)}s>\n<image>"
        if include_question and self.question:
            return f"{self.question}\n{time_tag}"
        return time_tag

def boundary_key(value: float) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".")
