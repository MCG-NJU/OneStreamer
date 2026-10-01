from __future__ import annotations

import os
import sys
import json
import time
from queue import Queue
from threading import Thread
from typing import List, Optional, TextIO

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


class VideoFrameExtractor:

    def __init__(self, video_path: str, target_fps: float = 1.0, prefetch: bool = True):
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        self.frame_interval = max(int(round(self.original_fps / target_fps)), 1)
        self.num_extracted_frames = int(self.duration * target_fps)

        self._prefetch = prefetch
        if prefetch:
            self._queue: "Queue[Optional[Image.Image]]" = Queue(maxsize=4)
            Thread(target=self._producer, daemon=True).start()
        else:
            self._iter = self._sequential_iter()

    def _sequential_iter(self):
        idx, extracted = 0, 0
        while extracted < self.num_extracted_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % self.frame_interval == 0:
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                extracted += 1
            idx += 1

    def _producer(self):
        for img in self._sequential_iter():
            self._queue.put(img)
        self._queue.put(None)

    def get_frame_at_round(self, round_num: int) -> Image.Image:
        img = self._queue.get() if self._prefetch else next(self._iter, None)
        if img is None:
            raise StopIteration(f"No more frames at round {round_num}")
        return img

    def get_total_rounds(self) -> int:
        return self.num_extracted_frames

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()


class Qwen3VLStreamEngine:

    _END_TOKENS = ("<|im_end|>", "<|endoftext|>")

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        debug: bool = False,
        debug_log_path: Optional[str] = None,
        debug_max_text_chars: int = 4000,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        min_pixels = min_pixels or int(os.environ.get("MIN_PIXELS", 3136))
        max_pixels = max_pixels or int(os.environ.get("MAX_PIXELS", 100352))

        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels,
        )
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.debug: bool = debug
        self.debug_max_text_chars: int = debug_max_text_chars
        self._debug_step: int = 0
        self._debug_fp: Optional[TextIO] = None
        if debug and debug_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(debug_log_path)), exist_ok=True)
            self._debug_fp = open(debug_log_path, "a", buffering=1)

    def _dbg(self, *args) -> None:
        if not self.debug:
            return
        msg = " ".join(str(a) for a in args)
        try:
            if self._debug_fp is not None:
                self._debug_fp.write(msg + "\n")
            else:
                print(msg, file=sys.stderr, flush=True)
        except Exception:
            pass

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + f"\n... <truncated {len(text) - max_chars} chars> ...\n" + text[-half:]

    @staticmethod
    def _to_chat_messages(
        messages: List[dict], images: List[Image.Image]
    ) -> List[dict]:
        image_iter = iter(images)
        converted = []
        for m in messages:
            role, content = m["role"], m["content"]
            if "<image>" not in content:
                converted.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            segments = content.split("<image>")
            text_parts, image_parts = [], []
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if seg:
                    text_parts.append({"type": "text", "text": seg})
                if idx < len(segments) - 1:
                    image_parts.append({"type": "image", "image": next(image_iter)})
            converted.append({"role": role, "content": image_parts + text_parts})
        return converted

    def _strip_end_tokens(self, text: str) -> str:
        for end_tok in self._END_TOKENS:
            while text.rstrip().endswith(end_tok):
                text = text.rstrip()[: -len(end_tok)]
        return text.strip()

    def infer(self, data: dict, max_tokens: int = 128, temperature: float = 0.0,
              top_p: float = 1.0, top_k: int = 0) -> str:
        import torch

        chat = self._to_chat_messages(data["messages"], data["images"])

        if self.debug:
            self._debug_step += 1
            self._dbg(f"\n{'=' * 80}\n[DEBUG step={self._debug_step}] messages 结构 "
                      f"(system/user/assistant, 合计 {len(chat)} 条, 图片 {len(data['images'])} 张)\n{'=' * 80}")
            for i, m in enumerate(chat):
                role = m["role"]
                content = m["content"]
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if c.get("type") == "image":
                            parts.append("<<IMAGE>>")
                        elif c.get("type") == "text":
                            parts.append(repr(c["text"]))
                        else:
                            parts.append(repr(c))
                    self._dbg(f"  [{i:02d}] {role}: {' | '.join(parts)}")
                else:
                    self._dbg(f"  [{i:02d}] {role}: {repr(content)}")

        if self.debug:
            try:
                prompt_str = self.processor.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False,
                )
                self._dbg(f"\n[DEBUG step={self._debug_step}] apply_chat_template(tokenize=False)"
                          f" 得到的最终 prompt:\n"
                          f"{'-' * 80}\n"
                          f"{self._truncate(prompt_str, self.debug_max_text_chars)}\n"
                          f"{'-' * 80}")
            except Exception as e:
                self._dbg(f"[DEBUG] apply_chat_template(tokenize=False) 失败: {e}")

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        if self.debug:
            try:
                tok = self.processor.tokenizer
                input_ids = inputs["input_ids"]
                self._dbg(f"[DEBUG step={self._debug_step}] input_ids.shape = {tuple(input_ids.shape)}")
                for k, v in inputs.items():
                    if hasattr(v, "shape"):
                        self._dbg(f"    - {k}.shape = {tuple(v.shape)}  dtype={getattr(v, 'dtype', None)}")
                ids = input_ids[0].tolist()
                head_ids, tail_ids = ids[:30], ids[-30:]
                self._dbg(f"    - head tokens: {tok.convert_ids_to_tokens(head_ids)}")
                self._dbg(f"    - tail tokens: {tok.convert_ids_to_tokens(tail_ids)}")
            except Exception as e:
                self._dbg(f"[DEBUG] 读取 inputs 元信息失败: {e}")

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
        else:
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 0

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=False)[0]

        if self.debug:
            self._dbg(f"[DEBUG step={self._debug_step}] raw generated (keep special tokens): "
                      f"{repr(text)}")
            cleaned = self._strip_end_tokens(text)
            self._dbg(f"[DEBUG step={self._debug_step}] final answer: {repr(cleaned)}")
            return cleaned

        return self._strip_end_tokens(text)


class StreamingSession:

    def __init__(
        self,
        engine: Qwen3VLStreamEngine,
        system: Optional[str] = SYSTEM,
        question: Optional[str] = None,
        question_time: int = 0,
        max_rounds: int = 32,
        global_question: bool = True,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        debug: bool = False,
    ):
        self.engine = engine
        self.system = system
        self.question = question
        self.question_time = question_time
        self.max_rounds = max_rounds
        self.global_question = global_question
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.debug = debug

        self._data: Optional[dict] = None
        self._last_answer: Optional[str] = None

    def _user_content(self, round_idx: int, include_question: bool) -> str:
        time_tag = f"<{round_idx}s-{round_idx + 1}s>\n<image>"
        if include_question and self.question:
            return f"{self.question}\n{time_tag}"
        return time_tag

    def _append_turn(self, frame: Image.Image, round_idx: int) -> None:
        if self._data is None:
            messages = []
            if self.system:
                messages.append({"role": "system", "content": self.system})
            include_q = self.global_question or (round_idx == self.question_time)
            messages.append({"role": "user", "content": self._user_content(round_idx, include_q)})
            self._data = {"images": [frame], "messages": messages}
            return

        messages = self._data["messages"]
        if self._last_answer is not None:
            messages.append({"role": "assistant", "content": self._last_answer})
        include_q = round_idx == self.question_time
        messages.append({"role": "user", "content": self._user_content(round_idx, include_q)})
        self._data["images"].append(frame)

        if len(self._data["images"]) > self.max_rounds:
            rounds_to_remove = len(self._data["images"]) - self.max_rounds
            start_round = round_idx - self.max_rounds + 1

            new_messages = messages[:1]
            new_messages.extend(messages[1 + rounds_to_remove * 2:])

            include_q_start = self.global_question or (self.question_time == start_round)
            new_messages[1] = {
                "role": "user",
                "content": self._user_content(start_round, include_q_start),
            }
            self._data["messages"] = new_messages
            self._data["images"] = self._data["images"][rounds_to_remove:]

    def step(self, frame: Image.Image, round_idx: int) -> str:
        self._append_turn(frame, round_idx)

        if self.debug:
            msgs = self._data["messages"] if self._data else []
            n_img = len(self._data["images"]) if self._data else 0
            self.engine._dbg(
                f"\n###### StreamingSession.step round_idx={round_idx} "
                f"question_time={self.question_time} global_question={self.global_question} "
                f"| turns={len(msgs)} images={n_img} ######"
            )

        if round_idx < self.question_time:
            answer = "</Silence>"
            if self.debug:
                self.engine._dbg(
                    f"  [round {round_idx}] 短路返回 '</Silence>' (未调用模型; "
                    f"round_idx < question_time={self.question_time})"
                )
        else:
            answer = self.engine.infer(
                self._data, max_tokens=self.max_tokens,
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            )
        self._last_answer = answer
        return answer

    def reset(self) -> None:
        self._data = None
        self._last_answer = None
