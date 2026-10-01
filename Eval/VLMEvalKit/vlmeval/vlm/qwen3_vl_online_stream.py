from __future__ import annotations

import os
import sys

from .streaming_video_base import (
    StreamingVideoModel,
    _format_time_tag,
    _round_span,
)

def _make_proactive_session_class(StreamingSession):

    class ProactiveStreamingSession(StreamingSession):
        def __init__(self, *args, extra_turns=None, target_fps=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self._extra_turns = list(extra_turns or [])
            self._target_fps = float(target_fps)

        def _user_content(self, round_idx: int, include_question: bool) -> str:
            base = super()._user_content(round_idx, include_question)
            old_tag = f'<{round_idx}s-{round_idx + 1}s>'
            base = base.replace(
                old_tag,
                _format_time_tag(round_idx, self._target_fps),
                1,
            )
            start, end = _round_span(round_idx, self._target_fps)
            hits = [
                t['content']
                for t in self._extra_turns
                if start < float(t['time']) <= end
                and t.get('content')
            ]
            if not hits:
                return base
            return '\n'.join(hits) + '\n' + base

    return ProactiveStreamingSession



class Qwen3VLOnlineStream(StreamingVideoModel):

    INSTALL_REQ = False

    def __init__(
        self,
        model_path: str,
        max_rounds: int = 32,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        target_fps: float = 1.0,
        global_question: bool = True,
        attn_implementation: str = 'flash_attention_2',
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.target_fps = target_fps
        self.global_question = bool(global_question)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        from shared.inference_fast import SYSTEM, Qwen3VLStreamEngine, StreamingSession, VideoFrameExtractor

        self._SYSTEM = SYSTEM if system_prompt is None else system_prompt
        self._VideoFrameExtractor = VideoFrameExtractor
        self._ProactiveStreamingSession = _make_proactive_session_class(
            StreamingSession
        )
        self._session = None

        print(
            f'[Qwen3VLOnlineStream] Loading engine from: {model_path}',
            flush=True,
        )
        self._engine = Qwen3VLStreamEngine(
            model_path,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )


    def reset_session(self, question: str, extra_turns: list):
        self._session = self._ProactiveStreamingSession(
            engine=self._engine,
            system=self._SYSTEM,
            question=question,
            question_time=0,
            max_rounds=self.max_rounds,
            global_question=self.global_question,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            extra_turns=extra_turns,
            target_fps=self.target_fps,
        )

    def step(self, frame, round_idx: int) -> str:
        assert self._session is not None, (
            'reset_session must be called before step'
        )
        return self._session.step(frame, round_idx=round_idx)

    def _open_extractor(self, video_path: str):
        return self._VideoFrameExtractor(video_path, target_fps=self.target_fps, prefetch=False)
