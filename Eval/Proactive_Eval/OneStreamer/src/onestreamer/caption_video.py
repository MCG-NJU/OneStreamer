from __future__ import annotations

import math
from dataclasses import dataclass

from .prompts import get_caption_control_protocol, validate_caption_control_tokens
from .video import iter_stream_frames


@dataclass
class VideoUpdate:
    start: float
    end: float
    frames: list
    timestamps: list[float]


def iter_video_updates(path, start, end, fps):
    frames, timestamps = [], []
    current_round = None
    for timestamp, _, frame in iter_stream_frames(path, start, end, fps):
        index = int(math.floor(timestamp - start + 1e-7))
        if current_round is not None and index != current_round:
            a = start + current_round
            yield a, min(a + 1, end), VideoUpdate(a, min(a + 1, end), frames, timestamps)
            frames, timestamps = [], []
        current_round = index
        frames.append(frame)
        timestamps.append(timestamp)
    if frames:
        a = start + current_round
        yield a, min(a + 1, end), VideoUpdate(a, min(a + 1, end), frames, timestamps)


class VideoCaptionSession:

    def __init__(self, engine, sample, config):
        self.engine = engine
        self.protocol = get_caption_control_protocol(config)
        validate_caption_control_tokens(engine.processor.tokenizer, self.protocol)
        self.fps = int(config['fps'])
        self.max_frames = int(config['context_frames'])
        if self.fps not in (2, 4) or self.max_frames % self.fps:
            raise ValueError('Video caption expects fps=2/4 and a whole number of seconds in context_frames')
        self.max_rounds = self.max_frames // self.fps
        self.min_pixels = int(config.get('min_pixels', 3136))
        self.max_pixels = int(config.get('max_pixels', 100352))
        self.max_tokens = int(config.get('max_new_tokens', 128))
        self.history = []
        self._data = None
        self._last_answer = None
        self.last_input_tokens = 0

    def encode(self, update):
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        frames = list(update.frames)
        timestamps = list(update.timestamps)
        temporal = int(self.engine.processor.video_processor.temporal_patch_size)
        while len(frames) % temporal:
            frames.append(frames[-1])
            timestamps.append(timestamps[-1])
        vt = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in frames])
        image_processor = self.engine.processor.image_processor
        factor = int(image_processor.patch_size) * int(image_processor.merge_size)
        height, width = smart_resize(
            vt.shape[-2], vt.shape[-1], factor=factor,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        vt = TF.resize(vt, [height, width], interpolation=TF.InterpolationMode.BICUBIC)
        vp = self.engine.processor.video_processor(
            videos=[vt], do_sample_frames=False, do_resize=False, return_tensors='pt')
        grid = vp['video_grid_thw'][0]
        merge = int(self.engine.processor.video_processor.merge_size)
        tokens_per_block = int(grid[1] * grid[2]) // (merge * merge)
        if int(grid[0]) != len(timestamps) // temporal:
            raise ValueError('Video processor temporal blocks disagree with sampled timestamps')
        block = '<|vision_start|>' + '<|video_pad|>' * tokens_per_block + '<|vision_end|>'
        visual = ''.join(
            f'<{sum(timestamps[i:i+temporal])/temporal:.1f} seconds>' + block
            for i in range(0, len(timestamps), temporal))
        visual += f'<{update.start:.1f}-{update.end:.1f} seconds>'
        return dict(visual=visual, tensors=vp, start=update.start, end=update.end,
                    raw_frames=len(update.frames), resized_wh=[width, height], answer=None)

    def build_inputs(self, stream_end=False):
        import torch

        chat = [{'role': 'system', 'content': self.protocol.system}]
        for i, turn in enumerate(self.history):
            text = (self.protocol.instruction.replace('<VIDEO_CONTEXT>', turn['visual'])
                    if i == 0 else turn['visual'])
            chat.append({'role': 'user', 'content': text})
            if turn['answer'] is not None:
                chat.append({'role': 'assistant', 'content': turn['answer']})
        if stream_end:
            chat.append({'role': 'user', 'content': '<STREAM_END>'})
        tokenizer = self.engine.processor.tokenizer
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt')
        for key in ('pixel_values_videos', 'video_grid_thw'):
            inputs[key] = torch.cat([t['tensors'][key] for t in self.history])
        self.last_input_tokens = inputs['input_ids'].shape[-1]
        return inputs, chat

    def generate(self, stream_end=False):
        import torch

        inputs, _ = self.build_inputs(stream_end)
        inputs = {k: v.to(self.engine.device) for k, v in inputs.items()}
        tokenizer = self.engine.processor.tokenizer
        with torch.inference_mode():
            out = self.engine.model.generate(
                **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                temperature=1.0, top_p=1.0, top_k=0, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = self.engine.processor.batch_decode(
            out[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=False)[0]
        return self.engine._strip_end_tokens(text)

    def step(self, update, round_idx):
        turn = self.encode(update)
        self.history.append(turn)
        self.history = self.history[-self.max_rounds:]
        self._data = {}
        self._last_answer = self.generate()
        turn['answer'] = self._last_answer
        return self._last_answer

    def finish_boundary(self):
        return self.generate(stream_end=True)
