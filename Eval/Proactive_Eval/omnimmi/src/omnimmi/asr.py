from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path

from .common import config_fingerprint, load_annotations, sample_id, source_video_path
from .models import build_offline
from .protocols import ap_prompt, build_visual_request, md_prompt, resolve_media_path, sg_prompt
from .sampling import materialize_frames, probe_video

S0 = 'You are a real-time streaming video analysis assistant. Please observe the live video stream and answer the question strictly based on the visual information.'
S1 = 'You are a real-time streaming video analysis assistant. Please observe the live video stream and answer the question strictly based on the visual information and any provided speech transcript available up to the question time.\nThe transcript is automatic and may contain errors. Use it as supporting evidence. Distinguish past events and descriptions of planned actions from the current visual state. If no transcript is provided, use the visual information only.'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=1200)
def load_words(root, video):
    path = Path(root) / 'segments' / (video + '.json')
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('schema_version') != 1 or data.get('video') != video or data.get('source_config_fingerprint') != 'fb52145bd53ff627':
        raise ValueError(f'Incompatible minimal word cache: {path}')
    return data, str(path), hashlib.sha256(raw).hexdigest()


def valid_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def select_words(data, cutoff, first_frame, tokenizer, budget=4096):
    if not valid_number(cutoff):
        raise ValueError('Nonfinite query time')
    candidates = []
    for i, w in enumerate(data['words']):
        a, b = w.get('start'), w.get('end')
        if not (valid_number(a) and valid_number(b) and 0 <= a <= b <= cutoff
                and isinstance(w.get('word'), str) and w['word'].strip()):
            continue
        if b < first_frame:
            continue
        candidates.append(dict(w, original_index=i))
    def count(ws):
        return len(tokenizer.encode(''.join(w['word'] for w in ws), add_special_tokens=False))
    offset = 0
    while count(candidates[offset:]) > budget:
        offset += 1
    words = candidates[offset:]
    units, current = [], []
    def emit():
        if current:
            units.append(dict(start=min(w['start'] for w in current), end=max(w['end'] for w in current),
                text=''.join(w['word'] for w in current).strip(),
                word_indices=[w['original_index'] for w in current]))
    for w in words:
        if current and (w['original_index'] != current[-1]['original_index'] + 1
                or len(current) >= 24 or w['start'] - current[-1]['end'] >= .6
                or max(w['end'], max(x['end'] for x in current)) - min(w['start'], min(x['start'] for x in current)) > 3):
            emit()
            current = []
        current.append(w)
        if re.search(r'[.!?。！？;；][\"\u0027”’）)]*$', w['word'].strip()):
            emit()
            current = []
    emit()
    return dict(candidate_indices=[w['original_index'] for w in candidates],
                word_indices=[w['original_index'] for w in words],
                budget_dropped_indices=[w['original_index'] for w in candidates[:offset]],
                lexical_tokens=count(words), units=units,
                quality=dict(zero_duration=sum(w['end'] == w['start'] for w in words),
                    long_words=sum(w['end'] - w['start'] > 3 for w in words),
                    end_regressions=sum(b['end'] < a['end'] for a,b in zip(words, words[1:]))))


def native_times(indices, fps, merge_size=2):
    if merge_size != 2 or not indices or not valid_number(fps) or fps <= 0:
        raise ValueError('Unsupported temporal processing')
    padded = list(indices)
    if len(padded) % 2:
        padded.append(padded[-1])
    centers = [(padded[i] / fps + padded[i+1] / fps) / 2 for i in range(0, len(padded), 2)]
    return padded, centers, [f'<{t:.1f} seconds>' for t in centers]


def render(selection):
    if not selection['word_indices']: return ''
    return '\n'.join(['ASR transcript (source-video timestamps):'] +
        [f"[{u['start']:.3f}-{u['end']:.3f}] {u['text']}" for u in selection['units']])

def make_message(paths, fps, prompt, transcript):
    content = [dict(type='text',value=transcript)] if transcript else []
    return content + [dict(type='video',value=paths,sample_fps=fps),dict(type='text',value=prompt)]


class AuditedProcessor:
    def __init__(self, processor, context_limit):
        self.original = processor
        self.context_limit = context_limit
        self.context = None
        self.audit = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        c = self.context
        if c is None or kwargs.get('do_sample_frames') is not False:
            raise ValueError('Expected explicit frame list without resampling')
        metadata = copy.deepcopy(kwargs['video_metadata'])
        if len(metadata) != 1 or len(kwargs['videos']) != 1:
            raise ValueError('Exactly one video is required')
        indices, centers, anchors = native_times(c['indices'], c['fps'], self.video_processor.merge_size)
        if len(kwargs['videos'][0]) != len(indices):
            raise ValueError('Unexpected video padding or frame resampling')
        metadata[0].update(fps=c['fps'], frames_indices=indices, total_num_frames=c['total_frames'])
        kwargs['video_metadata'] = metadata
        inputs = self.original(*args, **kwargs)
        ids = inputs['input_ids'][0]
        decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        actual = re.findall(r'(<[0-9.]+ seconds>)(?=<\|vision_start\|>)', decoded)
        if actual != anchors:
            raise ValueError(f'Native timestamp mismatch: {actual!r} != {anchors!r}')
        if len(ids) + c['max_new_tokens'] > self.context_limit:
            raise ValueError('Input plus generation exceeds context; no silent truncation permitted')
        self.audit = dict(mode=c['mode'], padded_source_indices=indices,
            native_centers=centers, native_timestamps=actual,
            input_tokens=len(ids), input_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            video_grid_thw=inputs['video_grid_thw'].tolist(), processor_video_metadata=metadata,
            source_frame_times=[i / c['fps'] for i in c['indices']])
        if c.get('hash_pixels'):
            pixels = inputs['pixel_values_videos'].contiguous()
            self.audit['pixel_sha256'] = hashlib.sha256(pixels.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
        return inputs


class Runner:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.tokenizer = None
        self.video_cache = {}
        self.selection_cache = {}

    def video(self, path):
        key = str(path)
        if key not in self.video_cache:
            self.video_cache[key] = probe_video(key)
        return self.video_cache[key]

    def prepare(self, config):
        if config["with_asr"] and self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config['budget_tokenizer'])
        if self.model_path != config['model_path']:
            if self.model is not None:
                del self.model
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            self.model = build_offline(config)
            mc = self.model.model.config
            limit = getattr(getattr(mc, 'text_config', mc), 'max_position_embeddings')
            self.model.processor = AuditedProcessor(self.model.processor, limit)
            self.model_path = config['model_path']
        self.model.system_prompt = config['system_prompt']
        self.model.generate_kwargs['max_new_tokens'] = config['max_new_tokens']

    def turn(self, config, task, sample, qa, prompt):
        start = time.monotonic()
        source = self.video(source_video_path(config['data_root'], sample['video']))
        media = self.video(resolve_media_path(config, task, sample, qa))
        request = build_visual_request(config=config, task=task, sample=sample,
            source_video=source, media_video=media, target_fps=4., recent_frames=32, qa=qa)
        paths, visual = materialize_frames(media, request.plan, config['cache_root'])
        indices = list(request.plan.indices)
        _, _, anchors = native_times(indices, source.source_fps)
        first = indices[0] / source.source_fps
        selected = None
        transcript = ''
        if config['with_asr']:
            data, asr_path, asr_hash = load_words(config['asr_root'], sample['video'])
            key = (asr_hash, request.semantic_target_seconds, first)
            if key not in self.selection_cache:
                self.selection_cache[key] = select_words(data, request.semantic_target_seconds, first, self.tokenizer)
            selected = copy.deepcopy(self.selection_cache[key])
            transcript = render(selected)
        context = dict(indices=indices, fps=source.source_fps, total_frames=source.total_frames,
            mode='source_absolute', max_new_tokens=config['max_new_tokens'], hash_pixels=False)
        self.model.processor.context = context
        message = make_message(paths, request.plan.sample_fps, prompt, transcript)
        import torch
        torch.cuda.reset_peak_memory_stats()
        prediction = str(self.model.generate(message=message, dataset='OmniMMI')).strip()
        peak_bytes = torch.cuda.max_memory_allocated()
        audit = self.model.processor.audit
        if audit is None:
            raise RuntimeError('Processor auditing did not run')
        visual.update(semantic_target_seconds=request.semantic_target_seconds, target_clamped=request.target_clamped,
            protocol=config['protocol'], input_is_official_clip=False, processor=audit)
        if selected is not None:
            selected.update(asr_path=asr_path, asr_sha256=asr_hash, cutoff=request.semantic_target_seconds,
                first_frame=first, rendered_text=transcript,
                input_word_indices=selected['word_indices'] if transcript else [],
                rendered_tokens=len(self.tokenizer.encode(transcript, add_special_tokens=False)),
                actual_rendered_tokens=len(self.model.processor.tokenizer.encode(transcript, add_special_tokens=False)))
        return dict(prompt=prompt, prediction=prediction, pred=prediction, visual_input=visual,
            asr_input=selected, elapsed_seconds=time.monotonic()-start, empty_answer=not prediction,
            peak_cuda_allocated_bytes=peak_bytes,
            output_tokens=len(self.model.processor.tokenizer.encode(prediction, add_special_tokens=False)))

    def sample(self, config, task, index):
        sample = load_annotations(config['data_root'], task)[index]
        result = copy.deepcopy(sample)
        result.update(sample_id=sample_id(task,index), task=task, suite=config['suite'], protocol=config['protocol'],
            evaluation_variant='32frames', target_fps=4., recent_frames=32,
            config_fingerprint=config_fingerprint(config), condition=config['condition'],
            system_sha256=digest(config['system_prompt']), model_path=config['model_path'])
        if task in ('ap','si'):
            prompt = ap_prompt(sample) if task == 'ap' else sample['question']
            result.update(self.turn(config,task,sample,None,prompt))
        else:
            previous, history = None, []
            for i, (qa, out) in enumerate(zip(sample['qa'], result['qa'])):
                prompt = md_prompt(qa['question'],previous) if task == 'md' else sg_prompt(history,qa['question'])
                out.update(self.turn(config,task,sample,qa,prompt), turn_index=i)
                previous = out['prediction']
                history.append((qa['question'],previous))
        return result
