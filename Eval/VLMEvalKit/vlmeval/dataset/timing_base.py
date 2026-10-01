import json
import os
import re
import string
import warnings
from collections import defaultdict

import pandas as pd
from PIL import Image

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset
from ..paths import dataset_root


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def _default_ovobench_root():
    return dataset_root('ovobench', 'OVOBENCH_ROOT')






def _has_value(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ''


def _float_token(value):
    if not _has_value(value):
        return 'na'
    return str(int(round(float(value) * 1000)))












class OVOBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='OVOBench', nframe=0, fps=-1, frames_limit=4096, min_pixels=28*28, max_pixels=448*448, total_pixels=65536*4*16*16):
        self.frames_limit = frames_limit
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)





    @staticmethod
    def _cache_key(task, video_rel_path, start, end, mode, source_kind):
        stem = str(video_rel_path).replace('\\', '/').rstrip('/')
        stem = stem.rsplit('.', 1)[0]
        stem = stem.replace('/', '__')
        return f'{task}__{stem}__{source_kind}__s{_float_token(start)}__e{_float_token(end)}__{mode}'

    @staticmethod
    def _candidate_media_paths(root_dir, task, video_rel_path):
        rel = str(video_rel_path).replace('\\', '/').strip().lstrip('/')
        candidates = []

        def add(path):
            if path not in candidates:
                candidates.append(path)

        task = str(task).strip()
        add(osp.join(root_dir, rel))
        add(osp.join(root_dir, 'chunked_videos', rel))
        add(osp.join(root_dir, task, rel))
        add(osp.join(root_dir, task, 'chunked_videos', rel))
        add(osp.join(root_dir, 'video', rel))
        add(osp.join(root_dir, 'video', 'chunked_videos', rel))

        base, ext = osp.splitext(rel)
        if not ext:
            for suffix in VIDEO_EXTENSIONS:
                add(osp.join(root_dir, rel + suffix))
                add(osp.join(root_dir, 'chunked_videos', rel + suffix))
                add(osp.join(root_dir, task, rel + suffix))
                add(osp.join(root_dir, task, 'chunked_videos', rel + suffix))
                add(osp.join(root_dir, 'video', rel + suffix))
                add(osp.join(root_dir, 'video', 'chunked_videos', rel + suffix))

        return candidates

    def _resolve_media_path(self, line):
        candidates = self._candidate_media_paths(self.data_root, line.get('task', ''), line['video'])
        for path in candidates:
            if osp.isfile(path):
                return path, 'file'
            if osp.isdir(path):
                return path, 'dir'
        raise FileNotFoundError(
            f'OVOBench media not found for task={line.get("task", "")}, video={line["video"]}. '
            f'Tried: {candidates}'
        )
