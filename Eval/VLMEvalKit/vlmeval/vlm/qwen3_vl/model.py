import os
import torch
from ..base import BaseModel
def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')

def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(BaseModel):
    VIDEO_LLM=True
    def __init__(self,model_path,min_pixels=None,max_pixels=None,total_pixels=None,max_new_tokens=4096,temperature=0.,top_p=1.,top_k=0,do_sample=False,repetition_penalty=1.,system_prompt=None,seed=42,torch_dtype='auto',**kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
        set_seed(seed)
        self.seed=seed; self.model_path=model_path
        self.min_pixels=min_pixels; self.max_pixels=max_pixels; self.total_pixels=total_pixels
        self.temperature=temperature; self.top_p=top_p; self.top_k=top_k; self.repetition_penalty=repetition_penalty
        self.max_new_tokens=max_new_tokens; self.system_prompt=system_prompt
        self.verbose = False
        self.fps=kwargs.get('fps',2);self.nframe=kwargs.get('nframe',128);self.FRAME_FACTOR=2
        self.generate_kwargs=dict(max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p,top_k=top_k,do_sample=do_sample,repetition_penalty=repetition_penalty)
        self.processor=AutoProcessor.from_pretrained(model_path)
        self.model=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch_dtype,device_map='auto',attn_implementation='flash_attention_2').eval()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if self.min_pixels is not None: item['min_pixels']=self.min_pixels
                if self.max_pixels is not None: item['max_pixels']=self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    @torch.inference_mode()
    def generate_inner_transformers(self,message,dataset=None):
        from qwen_vl_utils import process_vision_info
        messages=[]
        if self.system_prompt is not None: messages.append(dict(role='system',content=self.system_prompt))
        messages.append(dict(role='user',content=self._prepare_content(message,dataset)))
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos,video_kwargs=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
        metadata=None
        if videos is not None:
            videos,metadata=zip(*videos); videos,metadata=list(videos),list(metadata)
        inputs=self.processor(text=text,images=images,videos=videos,video_metadata=metadata,do_resize=False,return_tensors='pt',**(video_kwargs or {}))
        inputs=inputs.to(self.model.device).to(self.model.dtype)
        ids=self.model.generate(**inputs,**self.generate_kwargs)
        ids=[o[len(i):] for i,o in zip(inputs.input_ids,ids)]
        return self.processor.tokenizer.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    def generate_inner(self,message,dataset=None):
        return self.generate_inner_transformers(message,dataset)
