from __future__ import annotations

import json
import math
import os
import os.path as osp
import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..smp import LMUDataRoot, load, dump, get_logger
from ..smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset
from ..paths import PATHS, resolve_path

logger = get_logger('ProactiveVideoQA')


STREAM_META_PREFIX = '__PROACTIVE_STREAM_META__'

_DEFAULT_DATA_ROOT = PATHS['proactivevideoqa']
TARGET_FPS = 1.0

_TASK_MAP = {
    'ProactiveVideoQA_EGO': 'EGO',
    'ProactiveVideoQA_TV': 'TV',
    'ProactiveVideoQA_VAD': 'VAD',
    'ProactiveVideoQA_WEB': 'WEB',
}

_REQUIRED_COLS = frozenset({
    'index', 'question_id', 'video', 'question',
    'duration', 'task', 'conversation_json', 'answer_json',
})

JUDGE_INSTRUCTION = (
    'You are an evaluator for a video question answering system. '
    'Your task is to rate whether the predicted answer covers the key points '
    'of the ground truth answer. Use the following scale to assign a score:\n'
    '- 3: Mostly covered; the predicted answer covers all key information in '
    'the ground truth answer, though it may have minor inaccuracies or rephrases.\n'
    '- 2: Partially covered; the predicted answer has some correct information, '
    'but also contains significant inaccuracies or missing key points.\n'
    '- 1: Incorrect; the predicted answer may be related to the ground truth '
    'answer, but most of the information is missing, or the predicted answer is '
    'not relevant to the question or in very poor quality.\n\n'
    'Output the score only, do not add more explanations.'
)








def _compute_expected_rounds(duration: float, answer: list, fps: float = 1.0) -> int:
    last_reply_end = float(answer[-1]['reply_timespan'][1])
    by_frames = int(float(duration) * fps)
    by_reply = int(math.ceil(min(float(duration), last_reply_end) * fps))
    return max(1, min(by_frames, by_reply))






class ProactiveVideoQA(VideoBaseDataset):

    TYPE = 'Video-Proactive'
    MODALITY = 'VIDEO'

    @classmethod
    def supported_datasets(cls):
        return list(_TASK_MAP.keys())


    def prepare_dataset(self, dataset, root_dir=None):
        task = _TASK_MAP[dataset]
        data_root = root_dir or os.environ.get(
            'PROACTIVE_VIDEOQA_ROOT', _DEFAULT_DATA_ROOT
        )
        data_root = str(resolve_path(data_root))
        video_dir = osp.join(data_root, task, 'videos')
        anno_path = osp.join(data_root, task, 'anno.json')

        if not osp.isfile(anno_path):
            raise FileNotFoundError(
                f'ProactiveVideoQA annotation not found: {anno_path}. '
                'Set environment variable PROACTIVE_VIDEOQA_ROOT.'
            )
        if not osp.isdir(video_dir):
            raise FileNotFoundError(
                f'ProactiveVideoQA video directory not found: {video_dir}. '
                'Set environment variable PROACTIVE_VIDEOQA_ROOT.'
            )

        tsv_dir = osp.join(LMUDataRoot(), 'ProactiveVideoQA')
        os.makedirs(tsv_dir, exist_ok=True)
        tsv_path = osp.join(tsv_dir, f'{task}.tsv')

        if not self._tsv_valid(tsv_path):
            self._build_tsv(anno_path, task, tsv_path)

        if task in ('EGO', 'VAD') and os.environ.get('PRED_FORMAT', 'xlsx') == 'xlsx':
            logger.warning(
                f'ProactiveVideoQA_{task} has long videos (up to ~270 rounds). '
                'Prediction JSON may exceed xlsx cell limit (32767 chars). '
                'Recommended: set env var PRED_FORMAT=tsv before running.'
            )

        return {'root': video_dir, 'data_file': tsv_path}

    @staticmethod
    def _tsv_valid(tsv_path):
        if not osp.exists(tsv_path):
            return False
        try:
            data = load(tsv_path)
        except Exception:
            return False
        return _REQUIRED_COLS.issubset(set(data.columns))

    @staticmethod
    def _build_tsv(anno_path, task, tsv_path):
        with open(anno_path, encoding='utf-8') as f:
            records = json.load(f)
        rows = []
        for idx, rec in enumerate(records):
            conv = rec.get('conversation', [])
            question = conv[0]['content'] if conv else ''
            rows.append({
                'index': idx,
                'question_id': str(rec['question_id']),
                'video': str(rec['video']),
                'question': question,
                'duration': float(rec.get('duration', 0.0)),
                'task': task,
                'conversation_json': json.dumps(conv, ensure_ascii=False),
                'answer_json': json.dumps(rec['answer'], ensure_ascii=False),
            })
        dump(pd.DataFrame(rows), tsv_path)
        logger.info(f'Built ProactiveVideoQA TSV: {tsv_path} ({len(rows)} rows)')

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if hasattr(line, 'to_dict'):
            line = line.to_dict()

        video_path = osp.join(self.data_root, str(line['video']))
        question = str(line['question'])
        duration = float(line['duration'])
        conversation = json.loads(str(line['conversation_json']))
        answer = json.loads(str(line['answer_json']))
        qid = str(line['question_id'])
        task = str(line.get('task', ''))

        extra_turns = [
            {'time': float(c.get('time', 0)), 'content': c['content']}
            for c in conversation[1:]
            if c.get('role', 'user') == 'user' and c.get('content')
        ]

        target_fps = float(self.fps if getattr(self, 'fps', -1) > 0 else TARGET_FPS)
        target = _compute_expected_rounds(duration, answer, fps=target_fps)
        meta = {
            'video_path': video_path,
            'question': question,
            'extra_turns': extra_turns,
            'answer': answer,
            'duration': duration,
            'qid': qid,
            'task': task,
            'target_fps': target_fps,
            'target_rounds': target,
        }

        return [
            {'type': 'video', 'value': video_path},
            {
                'type': 'text',
                'value': STREAM_META_PREFIX + json.dumps(meta, ensure_ascii=False),
            },
        ]
