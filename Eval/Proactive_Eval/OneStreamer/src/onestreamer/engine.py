from __future__ import annotations

from typing import Any

from shared.inference_fast import Qwen3VLStreamEngine


RECENT_FRAME_INPUT_VERSION = "source_pixels_patch_pad_v1"


class OneStreamerEngine(Qwen3VLStreamEngine):

    def _pad_to_patch_grid(self, image):
        from PIL import ImageOps

        image_processor = self.processor.image_processor
        factor = int(image_processor.patch_size) * int(image_processor.merge_size)
        width, height = image.size
        padded_width = ((width + factor - 1) // factor) * factor
        padded_height = ((height + factor - 1) // factor) * factor
        if (padded_width, padded_height) == (width, height):
            return image
        return ImageOps.expand(
            image,
            border=(0, 0, padded_width - width, padded_height - height),
            fill=0,
        )

    def generate_chat(
        self,
        chat: list[dict[str, Any]],
        images: list,
        max_tokens: int,
        temperature: float = 0.0,
        do_resize: bool = False,
    ) -> str:
        import torch

        prompt = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        processor_images = images
        if images and not do_resize:
            processor_images = [self._pad_to_patch_grid(image) for image in images]
        inputs = self.processor(
            text=[prompt],
            images=processor_images or None,
            padding=True,
            do_resize=do_resize,
            return_tensors="pt",
        )
        token_ids = inputs["input_ids"]
        visual_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        visual_tokens = int((token_ids == visual_id).sum().item())
        self.last_input_metrics = {
            "input_tokens": int(token_ids.shape[-1]),
            "visual_tokens": visual_tokens,
            "text_tokens": int(token_ids.shape[-1]) - visual_tokens,
            "input_image_sizes": [list(im.size) for im in (processor_images or [])],
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
        }
        inputs = inputs.to(self.device)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for key in ("pixel_values", "pixel_values_videos"):
                if key in inputs:
                    inputs[key] = inputs[key].to(model_dtype)

        generation = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation["temperature"] = temperature
        else:
            generation.update(temperature=1.0, top_p=1.0, top_k=0)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[-1]
        text = self.processor.batch_decode(
            output[:, prompt_length:], skip_special_tokens=False
        )[0]
        return self._strip_end_tokens(text)

    def answer(
        self,
        system: str,
        memory_text: str,
        recent_frames: list[tuple[float, Any]],
        question_text: str,
        max_tokens: int = 16,
        recent_resize_policy: str = "source_pixels",
    ) -> str:
        content: list[dict[str, Any]] = []
        if memory_text:
            content.append({"type": "text", "text": memory_text})
        else:
            content.append(
                {
                    "type": "text",
                    "text": "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>",
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "<RECENT_VISUAL_CONTEXT>\n"
                    "The following frames are ordered from oldest to newest. "
                    + ("They retain their decoded source pixels; only minimal border padding "
                       "may be added to align them to the model patch grid."
                       if recent_resize_policy == "source_pixels" else
                       "They have been resized while preserving aspect ratio.")
                ),
            }
        )
        images = []
        for timestamp, image in recent_frames:
            content.append({"type": "text", "text": f"<{timestamp:.3f}s>"})
            content.append({"type": "image", "image": image})
            images.append(image)
        content.append({"type": "text", "text": "</RECENT_VISUAL_CONTEXT>\n" + question_text})
        chat = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        return self.generate_chat(
            chat=chat,
            images=images,
            max_tokens=max_tokens,
            temperature=0.0,
            do_resize=False,
        )
