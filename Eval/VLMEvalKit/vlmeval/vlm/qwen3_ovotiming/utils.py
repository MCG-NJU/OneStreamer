from __future__ import annotations

import math
import torch
from PIL import Image
import os
from typing import Iterable


def _positive_int(value, default: int) -> int:
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def processor_visual_geometry(processor) -> tuple[int, int, int]:
    image_processor = getattr(processor, 'image_processor', None)
    patch_size = _positive_int(getattr(image_processor, 'patch_size', None), 16)
    merge_size = _positive_int(
        getattr(
            image_processor,
            'merge_size',
            getattr(image_processor, 'spatial_merge_size', None),
        ),
        2,
    )
    return patch_size, merge_size, patch_size * merge_size

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    force_resize: bool = False,
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    if force_resize:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
        return h_bar, w_bar
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


VLLM_MAX_IMAGE_INPUT_NUM = 24

STREAM_OUTPUT_TRIGGER_TAGS: tuple[str, ...] = ("</Standby>", "</Silence>")




def strip_file_url(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://") :]
    return path










def smart_video_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 28,
    frame_min_pixels: int = 16 * 28 * 28 * 4,
    frame_max_pixels: int = 1024 * 28 * 28 * 4,
    force_resize: bool = False,
):
    assert temporal_factor == 1, "OVO Timing image-stream inference requires temporal_factor=1"
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar, w_bar = smart_resize(
        height, width, factor, frame_min_pixels, frame_max_pixels, force_resize=force_resize
    )


    return h_bar, w_bar

class _Qwen3VLInferenceFastEngine:

    _END_TOKENS = ('<|im_end|>', '<|endoftext|>')

    def __init__(self, vlm: 'Qwen3VLChat') -> None:
        self.processor = vlm.processor
        self.model = vlm.model
        self.generation_options = dict(top_p=vlm.top_p, top_k=vlm.top_k, repetition_penalty=vlm.repetition_penalty)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _to_chat_messages(messages: list[dict], images: list[Image.Image]) -> list[dict]:
        image_iter = iter(images)
        converted = []
        for m in messages:
            role, content = m['role'], m['content']
            if '<image>' not in content:
                converted.append({'role': role, 'content': [{'type': 'text', 'text': content}]})
                continue
            segments = content.split('<image>')
            parts: list[dict] = []
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if seg:
                    parts.append({'type': 'text', 'text': seg})
                if idx < len(segments) - 1:
                    parts.append({'type': 'image', 'image': next(image_iter)})
            converted.append({'role': role, 'content': parts})
        return converted

    def _strip_end_tokens(self, text: str) -> str:
        for end_tok in self._END_TOKENS:
            while text.rstrip().endswith(end_tok):
                text = text.rstrip()[: -len(end_tok)]
        return text.strip()

    def infer(self, data: dict, max_tokens: int = 128, temperature: float = 0.0) -> str:
        chat = self._to_chat_messages(data['messages'], data['images'])



        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs['temperature'] = temperature

        gen_kwargs.update(self.generation_options)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs['input_ids'].shape[-1]
        new_tokens = out[:, prompt_len:]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=False)[0]
        return self._strip_end_tokens(text)








def extract_ovo_online_meta(message: list) -> dict | None:
    for item in message:
        if isinstance(item, dict) and item.get('ovo_online_meta'):
            return item['ovo_online_meta']
    return None


def max_num_frames_from_ovo_meta(meta: dict) -> int:
    return max(1, int(meta.get('max_num_frames', meta.get('max_rounds', 32))))
