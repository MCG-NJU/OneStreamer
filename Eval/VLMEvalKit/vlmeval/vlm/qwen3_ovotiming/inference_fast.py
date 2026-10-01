from __future__ import annotations

import os
import json
import time
from queue import Queue
from threading import Thread
from typing import List, Optional, Sequence, Tuple

import cv2
from PIL import Image


SYSTEM = """
You are a helpful assistant specializing in streaming video analysis. 
You will receive input frame by frame, each labeled with absolute time intervals 
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.  

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.  

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.  

Do not provide partial answers or speculate beyond the given information.  
Whenever you deliver an answer, begin with </Response>.
"""






class StreamingSession:

    def __init__(
        self,
        engine: Qwen3VLStreamEngine,
        system: Optional[str] = SYSTEM,
        question: Optional[str] = None,
        question_messages: Optional[List[dict]] = None,
        question_time: int = 0,
        max_num_frames: int = 120,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        print_freq: int = 50,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_messages = question_messages
        self.question_time = question_time
        self.max_num_frames = max_num_frames
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.print_freq = print_freq
        self.print_count = 0

        self._data: Optional[dict] = None
        self._last_answer: Optional[str] = None
        self._interval_by_step: dict[int, Optional[Tuple[float, float]]] = {}
        self._turn_image_counts: List[int] = []
        self._turn_meta: List[dict] = []
        self._last_injected_question_sig: Optional[tuple] = None

    def _question_signature(self) -> Optional[tuple]:
        if self.question_messages:
            items = []
            for m in self.question_messages:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", ""))
                content = str(m.get("content", ""))
                items.append((role, content))
            return ("messages", tuple(items)) if items else None
        if self.question:
            return ("text", str(self.question))
        return None

    def _resize_stream_frame(self, frame: Image.Image) -> Image.Image:
        print(frame.size, flush=True)
        out = frame.resize((336, 224))
        print(out.size, flush=True)
        return out

    def _user_content(
        self,
        round_idx: int,
        time_sec: Optional[Tuple[float, float]] = None,
    ) -> str:
        if time_sec is not None:
            lo, hi = time_sec
            time_tag = f"<{lo:.3f}s-{hi:.3f}s>\n<image>"
        else:
            time_tag = f"<{round_idx}s-{round_idx + 1}s>\n<image>"
        return time_tag

    def _user_content_multi(
        self,
        round_idx: int,
        time_secs: Sequence[Tuple[float, float]],
    ) -> str:
        parts: List[str] = []
        for lo, hi in time_secs:
            parts.append(f"<{lo:.3f}s-{hi:.3f}s>\n<image>")
        return "\n".join(parts)

    def _append_turn(
        self,
        frame: Image.Image,
        round_idx: int,
        time_sec: Optional[Tuple[float, float]] = None,
    ) -> None:
        if time_sec is None:
            ts_list = [(float(round_idx), float(round_idx + 1))]
        else:
            ts_list = [time_sec]
        self._append_turn_batch([frame], ts_list, round_idx)

    def _append_turn_batch(
        self,
        frames: Sequence[Image.Image],
        time_secs: Sequence[Tuple[float, float]],
        round_idx: int,
    ) -> None:
        if len(frames) != len(time_secs) or not frames:
            raise ValueError("frames and time_secs must be non-empty and the same length")

        self._interval_by_step[round_idx] = (time_secs[0][0], time_secs[-1][1])
        resized = frames

        if self._data is None:
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            include_q = self.global_question or (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append(
                {
                    "role": "user",
                    "content": self._user_content_multi(round_idx, time_secs),
                }
            )
            self._data = {"images": list(resized), "messages": messages}
            self._turn_image_counts = [len(resized)]
            self._turn_meta = [{"round_idx": round_idx, "time_secs": [tuple(t) for t in time_secs]}]
        else:
            messages = self._data["messages"]
            if self._last_answer is not None:
                messages.append({"role": "assistant", "content": self._last_answer})
            include_q = (not self.global_question) and (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append(
                {
                    "role": "user",
                    "content": self._user_content_multi(round_idx, time_secs),
                }
            )
            self._data["images"].extend(resized)
            self._turn_image_counts.append(len(resized))
            self._turn_meta.append({"round_idx": round_idx, "time_secs": [tuple(t) for t in time_secs]})


        self._trim_sliding_window()

    def _first_frame_user_message_index(self) -> int:
        m = self._data["messages"]
        for i, msg in enumerate(m):
            if msg.get("role") == "user" and "<image>" in str(msg.get("content", "")):
                return i
        return len(m)

    def _trim_sliding_window(self) -> None:
        if self._data is None:
            return
        images = self._data["images"]
        messages = self._data["messages"]


        while len(images) > self.max_num_frames:
            first_u = self._first_frame_user_message_index()
            if first_u + 1 >= len(messages):
                break
            n0 = self._turn_image_counts[0]
            rid = self._turn_meta[0]["round_idx"]
            self._interval_by_step.pop(rid, None)
            del messages[first_u : first_u + 2]
            self._data["images"] = images[n0:]
            images = self._data["images"]
            self._turn_image_counts.pop(0)
            self._turn_meta.pop(0)
            first_u = self._first_frame_user_message_index()
            if first_u < len(messages) and messages[first_u].get("role") == "user":
                tm0 = self._turn_meta[0]
                r = tm0["round_idx"]
                ts = tm0["time_secs"]
                messages[first_u]["content"] = self._user_content_multi(r, ts)

    def step(
        self,
        frame: Image.Image,
        round_idx: int,
        time_sec: Optional[Tuple[float, float]] = None,
    ) -> Optional[str]:
        self._append_turn(frame, round_idx, time_sec)

        answer = self.engine.infer(
            self._data, max_tokens=self.max_tokens, temperature=self.temperature,
        )
        self._last_answer = answer
        return answer

    def step_frames(
        self,
        frames: Sequence[Image.Image],
        time_secs: Sequence[Tuple[float, float]],
        round_idx: int,
    ) -> Optional[str]:
        self._append_turn_batch(list(frames), time_secs, round_idx)

        if round_idx < self.question_time:
            answer = None
        else:
            answer = self.engine.infer(
                self._data, max_tokens=self.max_tokens, temperature=self.temperature,
            )
        self._last_answer = answer
        return answer

    @property
    def question_message_index(self) -> Optional[int]:
        return None

    def reset(self) -> None:
        self._data = None
        self._last_answer = None
        self._interval_by_step = {}
        self._turn_image_counts = []
        self._turn_meta = []
        self._last_injected_question_sig = None


class StreamingSessionV3:

    def __init__(
        self,
        engine,
        system: Optional[str] = SYSTEM,
        question: Optional[str] = None,
        question_messages: Optional[List[dict]] = None,
        question_time: int = 0,
        max_num_frames: int = 120,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        print_freq: int = 50,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_messages = question_messages
        self.question_time = question_time
        self.max_num_frames = max_num_frames
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.print_freq = print_freq
        self.print_count = 0

        self._data: Optional[dict] = None
        self._last_answer: Optional[str] = None
        self._interval_by_step: dict[int, Optional[Tuple[float, float]]] = {}
        self._turn_video_counts: List[int] = []
        self._turn_meta: List[dict] = []
        self._last_injected_question_sig: Optional[tuple] = None

    def _question_signature(self) -> Optional[tuple]:
        if self.question_messages:
            items = []
            for m in self.question_messages:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", ""))
                content = str(m.get("content", ""))
                items.append((role, content))
            return ("messages", tuple(items)) if items else None
        if self.question:
            return ("text", str(self.question))
        return None

    @staticmethod
    def _user_content_video() -> str:
        return "<video>"

    def _append_turn_video(
        self,
        video_tensor,
        video_metadata: dict,
        round_idx: int,
        time_sec: Tuple[float, float],
    ) -> None:
        self._interval_by_step[round_idx] = time_sec

        if self._data is None:
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            include_q = self.global_question or (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append({"role": "user", "content": self._user_content_video()})
            self._data = {
                "videos": [video_tensor],
                "video_metadata": [video_metadata],
                "messages": messages,
            }
            self._turn_video_counts = [1]
            self._turn_meta = [{"round_idx": round_idx, "time_sec": tuple(time_sec)}]
        else:
            messages = self._data["messages"]
            if self._last_answer is not None:
                messages.append({"role": "assistant", "content": self._last_answer})
            include_q = (not self.global_question) and (round_idx == self.question_time)
            if include_q:
                sig = self._question_signature()
                if sig is not None and sig != self._last_injected_question_sig:
                    if self.question_messages:
                        messages.extend(self.question_messages)
                    elif self.question:
                        messages.append({"role": "user", "content": self.question})
                    self._last_injected_question_sig = sig
            messages.append({"role": "user", "content": self._user_content_video()})
            self._data["videos"].append(video_tensor)
            self._data["video_metadata"].append(video_metadata)
            self._turn_video_counts.append(1)
            self._turn_meta.append({"round_idx": round_idx, "time_sec": tuple(time_sec)})

        self._trim_sliding_window()

    def _first_video_user_message_index(self) -> int:
        m = self._data["messages"]
        for i, msg in enumerate(m):
            if msg.get("role") == "user" and "<video>" in str(msg.get("content", "")):
                return i
        return len(m)

    def _trim_sliding_window(self) -> None:
        if self._data is None:
            return
        videos = self._data["videos"]
        video_metadata = self._data["video_metadata"]
        messages = self._data["messages"]

        while len(videos) > self.max_num_frames:
            first_u = self._first_video_user_message_index()
            if first_u + 1 >= len(messages):
                break
            n0 = self._turn_video_counts[0]
            rid = self._turn_meta[0]["round_idx"]
            self._interval_by_step.pop(rid, None)
            del messages[first_u : first_u + 2]
            self._data["videos"] = videos[n0:]
            self._data["video_metadata"] = video_metadata[n0:]
            videos = self._data["videos"]
            video_metadata = self._data["video_metadata"]
            self._turn_video_counts.pop(0)
            self._turn_meta.pop(0)

    def step_clip(
        self,
        frames: Sequence[Image.Image],
        t_lo: float,
        t_hi: float,
        round_idx: int,
        clip_fps: Optional[float] = None,
    ) -> Optional[str]:
        if not frames:
            raise ValueError("frames must be non-empty")
        video_tensor = self.engine.pil_frames_to_video_tensor(list(frames))
        meta = self.engine.build_clip_metadata(list(frames), t_lo, t_hi, clip_fps=clip_fps)
        self._append_turn_video(video_tensor, meta, round_idx, (float(t_lo), float(t_hi)))

        answer = self.engine.infer_video(
            self._data, max_tokens=self.max_tokens, temperature=self.temperature,
        )
        self._last_answer = answer
        return answer

    def reset(self) -> None:
        self._data = None
        self._last_answer = None
        self._interval_by_step = {}
        self._turn_video_counts = []
        self._turn_meta = []
        self._last_injected_question_sig = None
