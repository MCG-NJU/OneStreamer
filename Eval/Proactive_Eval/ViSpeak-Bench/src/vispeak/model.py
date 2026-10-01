from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass


CONTROL_PATTERN = re.compile(
    r"^\s*(</(?:Silence|Standby|Response)>)", re.IGNORECASE
)


@dataclass(frozen=True)
class OnlineDecision:
    kind: str
    raw_output: str
    response_text: str | None = None


def parse_online_decision(output):
    raw_output = "" if output is None else str(output).strip()
    match = CONTROL_PATTERN.match(raw_output)
    if match is None:
        return OnlineDecision("unknown", raw_output)

    token = match.group(1).lower()
    if token == "</silence>":
        return OnlineDecision("silence", raw_output)
    if token == "</standby>":
        return OnlineDecision("standby", raw_output)

    response_text = raw_output[match.end():].strip()
    return OnlineDecision("response", raw_output, response_text)




class Qwen3VLOnline:

    def __init__(
        self,
        model_path,
        inference_file=None,
        device="cuda",
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        min_pixels=3136,
        max_pixels=100352,
        video_fps=1.0,
        max_rounds=32,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        debug=False,
        debug_log_path=None,
    ):
        if video_fps != 1.0:
            raise ValueError(
                "The current Qwen3-VL Online time-tag protocol requires video_fps=1.0"
            )

        from shared import inference_fast as online
        self.system_prompt = online.SYSTEM
        self.streaming_session_cls = online.StreamingSession
        self.frame_extractor_cls = online.VideoFrameExtractor
        self.video_fps = video_fps
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.engine = online.Qwen3VLStreamEngine(
            model_path=model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            debug=debug,
            debug_log_path=debug_log_path,
        )

    def create_session(self, task_prompt, start_time):
        question_time = int(math.floor(start_time))
        return self.streaming_session_cls(
            engine=self.engine,
            system=self.system_prompt,
            question=task_prompt,
            question_time=question_time,
            max_rounds=self.max_rounds,
            global_question=True,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            debug=getattr(self.engine, "debug", False),
        )

    def iter_frames(self, video_path, start_time, max_response_time):
        extractor = self.frame_extractor_cls(
            video_path, target_fps=self.video_fps, prefetch=False
        )
        try:
            for round_idx in range(extractor.get_total_rounds()):
                frame = extractor.get_frame_at_round(round_idx)
                response_time = round_idx + 1
                if response_time <= start_time:
                    continue
                if response_time > max_response_time:
                    break
                yield frame, round_idx, float(response_time)
        finally:
            extractor.close()

    @staticmethod
    def step(session, frame, round_idx):
        return parse_online_decision(session.step(frame, round_idx))

    def name(self):
        return "Qwen3-VL-Online"
