import json
import hashlib
import math
import os
import re
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ..smp import LMUDataRoot, dump, get_file_extension, load, osp
from .timing_base import OVOBench, VIDEO_EXTENSIONS, _default_ovobench_root


def _default_ovobench_online_json():
    return os.environ.get(
        'OVOBENCH_ONLINE_JSON',
        osp.join(
            osp.dirname(osp.dirname(osp.dirname(__file__))),
            'json_data',
            'ovobench',
            'ovobench_timing.json',
        ),
    ).strip()



STREAMING_SYS_CONSERVATIVE_THREE_PROTOCOL = """You are a conservative assistant for real-time streaming video event detection.
You receive the video incrementally, with every update labeled by an absolute time interval such as <0s-1s>. Base each decision only on evidence observed so far.

Every </Response> is a final detection of one distinct event boundary. An unmatched, premature, or duplicate response is an error, so prioritize precision over frequent responses.

At every update, use exactly one of these three protocols:

1. Use </Silence> when:
   - No relevant event is currently supported by clear visual evidence.
   - The requested event has not started, or a suspected event is no longer supported.
   - The same event was already reported and no unmistakably new occurrence or step has completed.
   - The scene is unchanged, shows only aftermath, or merely repeats evidence already used for a response.
   Output exactly </Silence> and nothing else.

2. Use </Standby> only when:
   - A specific relevant event is visibly developing, but its completion or identity still needs confirmation.
   - There is concrete new evidence, not just generic activity or uncertainty.
   </Standby> is reversible. Return to </Silence> if later frames do not confirm a new event.
   Output exactly </Standby> and nothing else.

3. Use </Response> only when:
   - The visual evidence clearly establishes that a new, distinct requested event or step has just completed.
   - All essential entities, actions, attributes, and relationships required by the question are confirmed.
   - This event has not already been reported.
   Output </Response> followed by a concise answer.

Task-specific rules:
- For a single-event question, respond at most once in the entire stream. After responding, always use </Silence>.
- For "How many times" questions, respond exactly once when each genuinely new occurrence completes. Do not repeat the same count while no new occurrence has happened.
- For step-description questions, respond exactly once when each genuinely new requested step completes. Do not repeatedly report an ongoing or already completed step.

Do not predict future actions, treat a precursor as completion, or use </Response> merely because the current frames are relevant. When uncertain, prefer </Standby> or </Silence>.
"""







def validate_timing_prediction(raw, row, expected_config=None):
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError('prediction must be a JSON object')
        if obj['sample_id'] != int(row['sample_id']) or obj['task'] != str(row['task']):
            raise ValueError('prediction sample_id/task mismatch')
        gt = row['gt_timestamp']
        gt = json.loads(gt) if isinstance(gt, str) else list(gt)
        if obj['gt_timestamp'] != gt:
            raise ValueError('prediction GT mismatch')
        for key in ('detections', 'checked_detections', 'response_events'):
            if not isinstance(obj[key], list):
                raise ValueError(f'{key} must be a list')
        times = obj['detections']
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
               for t in times):
            raise ValueError('non-finite or non-numeric detection timestamp')
        if times != [x['time'] for x in obj['checked_detections']]:
            raise ValueError('checked_detections/detections mismatch')
        if times != [x['time'] for x in obj['response_events']]:
            raise ValueError('response_events/detections mismatch')
        if expected_config is not None and obj.get('inference_config') != expected_config:
            raise ValueError('actual question/system/window/fps differs from the experiment')
        return obj
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"Invalid OVO prediction for sample {row['sample_id']}: {exc}") from exc




class OVOTiming(OVOBench):
    TYPE = 'Video-Streaming'
    MODALITY = 'VIDEO'

    def __init__(
        self,
        dataset='OVO_Timing',
        nframe=0,
        fps=4.0,
        streaming_fps: float | None = None,
        frames_limit: int = 4096,
        max_num_frames: int = 32,
        check_extracted_frames: bool = True,
    ):
        env_max_num_frames = os.environ.get('OVOTIMING_MAX_NUM_FRAMES', '').strip()
        if env_max_num_frames:
            max_num_frames = int(env_max_num_frames)
        if int(max_num_frames) <= 0:
            raise ValueError(f'max_num_frames must be positive, got {max_num_frames}')
        self.streaming_fps = float(streaming_fps) if streaming_fps is not None else float(fps)
        self.max_num_frames = int(max_num_frames)
        self.strict_predictions = os.environ.get('OVOTIMING_STRICT', '0') == '1'
        if self.strict_predictions:
            _streaming_system_prompt()
        task_filter_raw = os.environ.get('OVOTIMING_TASK_FILTER', '').strip()
        self.task_filter = tuple(
            dict.fromkeys(x.strip().upper() for x in task_filter_raw.split(',') if x.strip())
        )
        invalid_tasks = sorted(set(self.task_filter) - {'CRR', 'REC', 'SSR'})
        if invalid_tasks:
            raise ValueError(f'Invalid OVOTIMING_TASK_FILTER tasks: {invalid_tasks}')
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, frames_limit=frames_limit)
        if self.task_filter:
            self.data = self.data[self.data['task'].isin(self.task_filter)].copy().reset_index(drop=True)
            self.videos = sorted(set(self.data['video']))
        expected_samples = os.environ.get('OVOTIMING_EXPECTED_SAMPLES', '').strip()
        if expected_samples and len(self.data) != int(expected_samples):
            raise ValueError(
                f'OVO Timing sample count mismatch: expected {expected_samples}, got {len(self.data)}; '
                f'task_filter={self.task_filter or "ALL"}'
            )
        if self.strict_predictions:
            self.data = self.data.copy()
            self.data['effective_question'] = self.data.apply(self.effective_question, axis=1)


    def expected_inference_config(self, line):
        return {
            'question': self.effective_question(line),
            'system_sha256': hashlib.sha256(getattr(self, 'scoring_system_prompt', _streaming_system_prompt()).encode()).hexdigest(),
            'max_num_frames': self.max_num_frames,
            'streaming_fps': self.streaming_fps,
        }

    @classmethod
    def supported_datasets(cls):
        return ['OVO_Timing']

    @staticmethod
    def _gt_timestamps(record: dict[str, Any]) -> list[float]:
        task = record['task']
        if task == 'CRR':
            return [float(record['clue_time'])]
        if task == 'REC':
            return [float(t) + 2.0 for t in record['start_times']]
        return [float(t) + 2.0 for t in record['start_time']]

    @staticmethod
    def _task_question(record: dict[str, Any]) -> str:
        task = record['task']
        if task == 'REC':
            return 'How many times does the event: {} happen?'.format(record['activity'])
        if task == 'SSR':
            return 'Describe the steps of {}'.format(
                re.sub(r'(?<!^)(?=[A-Z])', ' ', record['tutorial']).lower()
            )
        return str(record['question'])

    @staticmethod
    def _end_time(record: dict[str, Any]) -> float:
        info = record['test_info']
        return float(info[-1]['realtime'])

    @classmethod
    def _build_tsv(cls, json_path: str, tsv_path: str, root_dir: str) -> None:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        rows = []
        index = 0
        for record in raw:
            task = record.get('task')
            if task not in ('REC', 'SSR', 'CRR'):
                continue
            sample_id = int(record.get('id', index))
            video_rel = f'{sample_id}.mp4'

            q = cls._task_question(record)
            gt_ts = cls._gt_timestamps(record)
            ask_t = record.get('ask_time')
            rows.append(
                dict(
                    index=index,
                    sample_id=sample_id,
                    task=task,
                    video=video_rel,
                    question=q,
                    answer=str(record.get('answer', '')),
                    gt_timestamp=json.dumps(gt_ts),
                    ask_time=ask_t if ask_t is not None else np.nan,
                    test_info=json.dumps(record['test_info']),
                    start_time=json.dumps(record['start_time']) if task == 'SSR' else '',
                    activity=str(record.get('activity', '')),
                    tutorial=str(record.get('tutorial', '')),
                    end_time=cls._end_time(record),
                )
            )
            index += 1

        dump(pd.DataFrame(rows), tsv_path)

    def prepare_dataset(self, dataset='OVO_Timing', root_dir=None, online_json=None):
        root_dir = root_dir if root_dir is not None else _default_ovobench_root()
        json_path = online_json if online_json is not None else _default_ovobench_online_json()
        tsv_path = osp.join(LMUDataRoot(), dataset + '.tsv')


        if not osp.isdir(root_dir):
            raise FileNotFoundError(
                f'OVO_Timing root directory not found: {root_dir}. Set OVOBENCH_ROOT.'
            )
        if not osp.isfile(json_path) and not osp.isfile(tsv_path):
            raise FileNotFoundError(f'OVO_Timing online json not found: {json_path}')

        if not osp.exists(tsv_path):
            self._build_tsv(json_path, tsv_path, root_dir)
        return dict(root=root_dir, data_file=tsv_path)

    @staticmethod
    def _id_search_roots(root_dir: str, task: str) -> list[str]:
        t = str(task).strip()
        roots = [root_dir]
        for sub in ('chunked_videos', t, osp.join(t, 'chunked_videos'), 'video', osp.join('video', 'chunked_videos')):
            roots.append(osp.join(root_dir, sub))
        seen: set[str] = set()
        out: list[str] = []
        for r in roots:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    @classmethod
    def _find_id_based_video_path(cls, root_dir: str, task: str, sample_id) -> str | None:
        if sample_id is None or (isinstance(sample_id, float) and pd.isna(sample_id)):
            return None
        sid = int(sample_id)
        prefix = f'{sid}_'
        best_path: str | None = None
        best_k = -1
        for base in cls._id_search_roots(root_dir, task):
            if not osp.isdir(base):
                continue
            try:
                names = os.listdir(base)
            except OSError:
                continue
            for name in names:
                stem, ext = osp.splitext(name)
                if ext.lower() not in VIDEO_EXTENSIONS:
                    continue
                if not stem.startswith(prefix):
                    continue
                suf = stem[len(prefix) :]
                if not suf.isdigit():
                    continue
                k = int(suf)
                path = osp.join(base, name)
                if osp.isfile(path) and k > best_k:
                    best_k = k
                    best_path = path
        if best_path is not None:
            return best_path
        rel = f'{sid}.mp4'
        for path in OVOBench._candidate_media_paths(root_dir, task, rel):
            if osp.isfile(path):
                return path
        return None

    def _resolve_media_path(self, line):
        sample_id = line.get('sample_id')
        path = self._find_id_based_video_path(self.data_root, str(line.get('task', '')), sample_id)
        if path is not None:
            return path, 'file'
        return super()._resolve_media_path(line)

    def save_forward_stream_frames(self, line) -> tuple[list[str], list[float]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        task = str(line['task'])
        test_info = json.loads(line['test_info']) if isinstance(line['test_info'], str) else line['test_info']
        end_time = float(test_info[-1]['realtime'])
        if task in ('REC', 'SSR'):
            start_time = 0.0
        else:
            start_time = float(line['ask_time'])

        dt = 1.0 / self.streaming_fps if self.streaming_fps > 0 else 0.25
        stream_times: list[float] = []
        curr = float(start_time)
        while curr <= end_time + 1e-6:
            stream_times.append(curr)
            curr += dt

        if not stream_times:
            raise ValueError('empty stream timeline')

        line_key = line if isinstance(line, pd.Series) else pd.Series(line)
        media_path, source_kind = self._resolve_media_path(line_key)

        if source_kind != 'file':
            raise NotImplementedError(
                f'OVO_Timing pre-extract requires a video file; got directory stream: {media_path}'
            )

        import decord
        from decord import cpu

        vr = decord.VideoReader(media_path, ctx=cpu(0), num_threads=1)
        source_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        if source_fps <= 0:
            raise ValueError(f'invalid video fps for {media_path}')

        frame_indices: list[int] = []
        for t in stream_times:
            fi = int(t * source_fps)
            fi = min(max(0, fi), max(0, total_frames - 1))
            frame_indices.append(fi)

        mode = f'forward_{str(self.streaming_fps).replace(".", "p")}fps'
        cache_key = self._cache_key(
            line_key.get('task', ''),
            line_key['video'],
            start_time,
            end_time,
            mode,
            source_kind,
        )
        cache_root = osp.join(self.frame_root, cache_key)
        os.makedirs(cache_root, exist_ok=True)
        n = len(stream_times)
        frame_paths = [osp.join(cache_root, f'stream-{i:06d}-of-{n}.jpg') for i in range(n)]

        pos_by_idx: dict[int, list[int]] = defaultdict(list)
        for pos, fi in enumerate(frame_indices):
            pos_by_idx[fi].append(pos)

        for fi in sorted(pos_by_idx.keys()):
            arr = vr[fi].asnumpy()
            im = Image.fromarray(arr)
            for pos in pos_by_idx[fi]:
                pth = frame_paths[pos]
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, stream_times

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]

        frame_paths, stream_times = self.save_forward_stream_frames(line)

        task = str(line['task'])
        test_info = json.loads(line['test_info']) if isinstance(line['test_info'], str) else line['test_info']
        end_time = float(test_info[-1]['realtime'])
        start_time = 0.0 if task in ('REC', 'SSR') else float(line['ask_time'])

        meta = {
            'frame_paths': frame_paths,
            'stream_times': stream_times,
            'streaming_fps': self.streaming_fps,
            'start_time': start_time,
            'end_time': end_time,
            'history_length': 5,
            'history_fps': 1,
            'proposal_update_duration': 9999,
            'max_num_frames': int(os.environ.get('OVOTIMING_MAX_NUM_FRAMES', self.max_num_frames)),
            'max_new_tokens': 128,
            'question': self.effective_question(line),
            'task': task,
            'sample_id': int(line['sample_id']) if not pd.isna(line.get('sample_id', np.nan)) else int(line['index']),
            'test_info': test_info,
            'gt_timestamp': json.loads(line['gt_timestamp'])
            if isinstance(line['gt_timestamp'], str)
            else list(line['gt_timestamp']),
            'prebuilt_frames': True,
        }
        if self.strict_predictions:
            meta['audit_inference_config'] = True
        return [
            dict(type='text', value=_streaming_system_prompt(), role='system'),
            dict(type='text', value='[OVOBench online streaming]', ovo_online_meta=meta),
        ]

    @staticmethod
    def _f1_stats(gt_times: list[float], det_times: list[float], tol: float = 2.0) -> tuple[float, float]:
        det_copy = list(det_times)
        correct = 0
        for gt in gt_times:
            for i, t in enumerate(det_copy):
                if abs(float(t) - float(gt)) <= tol:
                    correct += 1
                    det_copy.pop(i)
                    break
        recall = correct / len(gt_times) if gt_times else 0.0
        precision = correct / len(det_times) if det_times else 0.0
        return recall, precision

    @classmethod
    def for_scoring(cls, samples, protocol):
        from pathlib import PurePosixPath
        scorer = cls.__new__(cls)
        scorer.data = pd.DataFrame(samples).copy()
        scorer.data['sample_id'] = [int(PurePosixPath(row['video']).stem) for row in samples]
        scorer.strict_predictions = True
        scorer.streaming_fps = float(protocol['fps'])
        scorer.max_num_frames = int(protocol['max_num_frames'])
        scorer.scoring_system_prompt = protocol['system_prompt']
        return scorer

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'Evaluation file should be in xlsx/json/tsv format'
        )
        judge_name = judge_kwargs.get('model', 'exact_matching')
        if judge_name not in [None, 'exact_matching']:
            warnings.warn(f'OVO_Timing uses rule-based F1. Ignoring judge model `{judge_name}`.')

        from ..smp.file import get_intermediate_file_path

        score_file = get_intermediate_file_path(eval_file, '_score')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'json')

        data = load(eval_file)
        if self.strict_predictions:
            if len(data) != len(self.data) or data['sample_id'].duplicated().any():
                raise ValueError('OVO prediction rows are missing or duplicated')
            expected = self.data.set_index('sample_id', drop=False)
            if set(data['sample_id']) != set(expected.index):
                raise ValueError('OVO prediction sample IDs differ from the dataset')
            for _, row in data.iterrows():
                original = expected.loc[row['sample_id']]
                if row['task'] != original['task'] or json.loads(row['gt_timestamp']) != json.loads(original['gt_timestamp']):
                    raise ValueError('OVO result metadata differs from the dataset')
                validate_timing_prediction(row['prediction'], original, self.expected_inference_config(original))
        scored = data.copy()

        recalls_crr, precs_crr = [], []
        first_recalls_crr, first_precs_crr = [], []
        crr_response_counts = {'zero': 0, 'one': 0, 'multiple': 0}
        crr_first_timing = {'early': 0, 'hit': 0, 'late': 0, 'no_answer': 0}
        crr_first_hits = 0
        crr_answered = 0
        crr_total_responses = 0
        crr_raw_hits = 0
        recalls_rec, precs_rec = [], []
        recalls_ssr, precs_ssr = [], []
        parsed = []

        for _, row in scored.iterrows():
            pred_raw = row.get('prediction', '')
            gt_list = row.get('gt_timestamp', '[]')
            if isinstance(gt_list, str):
                gt_list = json.loads(gt_list)
            gt_list = [float(x) for x in gt_list]

            det_times: list[float] = []
            if isinstance(pred_raw, str) and pred_raw.strip().startswith('{'):
                try:
                    obj = json.loads(pred_raw)
                    if isinstance(obj.get('checked_detections'), list) and obj['checked_detections']:
                        det_times = [float(x['time']) for x in obj['checked_detections']]
                    elif isinstance(obj.get('detections'), list):
                        det_times = [float(x) for x in obj['detections']]
                    elif isinstance(obj.get('response_events'), list):
                        det_times = [float(x['time']) for x in obj['response_events']]
                except json.JSONDecodeError:
                    det_times = []
            parsed.append(json.dumps(det_times) if det_times else '')

            r, p = self._f1_stats(gt_list, det_times, tol=2.0)
            task = str(row.get('task', ''))
            if task == 'CRR':
                recalls_crr.append(r)
                precs_crr.append(p)
                n_response = len(det_times)
                crr_total_responses += n_response
                crr_raw_hits += int(round(p * n_response)) if n_response else 0
                if n_response == 0:
                    crr_response_counts['zero'] += 1
                    crr_first_timing['no_answer'] += 1
                    first_r, first_p = 0.0, 0.0
                else:
                    crr_answered += 1
                    crr_response_counts['one' if n_response == 1 else 'multiple'] += 1
                    first_time = det_times[0]
                    first_r, first_p = self._f1_stats(gt_list, [first_time], tol=2.0)
                    if first_r > 0:
                        crr_first_timing['hit'] += 1
                        crr_first_hits += 1
                    elif gt_list and first_time < gt_list[0] - 2.0:
                        crr_first_timing['early'] += 1
                    else:
                        crr_first_timing['late'] += 1
                first_recalls_crr.append(first_r)
                first_precs_crr.append(first_p)
            elif task == 'REC':
                recalls_rec.append(r)
                precs_rec.append(p)
            elif task == 'SSR':
                recalls_ssr.append(r)
                precs_ssr.append(p)

        scored['parsed_detections'] = parsed
        dump(scored, score_file)

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        rr_crr, pr_crr = _mean(recalls_crr), _mean(precs_crr)
        rr_rec, pr_rec = _mean(recalls_rec), _mean(precs_rec)
        rr_ssr, pr_ssr = _mean(recalls_ssr), _mean(precs_ssr)

        f1_crr = 2 * rr_crr * pr_crr / (rr_crr + pr_crr) if (rr_crr + pr_crr) > 0 else 0.0
        f1_rec = 2 * rr_rec * pr_rec / (rr_rec + pr_rec) if (rr_rec + pr_rec) > 0 else 0.0
        f1_ssr = 2 * rr_ssr * pr_ssr / (rr_ssr + pr_ssr) if (rr_ssr + pr_ssr) > 0 else 0.0
        present_task_metrics = [
            (rr_crr, pr_crr, f1_crr, len(recalls_crr)),
            (rr_rec, pr_rec, f1_rec, len(recalls_rec)),
            (rr_ssr, pr_ssr, f1_ssr, len(recalls_ssr)),
        ]
        present_task_metrics = [x for x in present_task_metrics if x[3] > 0]
        avg_rr = _mean([x[0] for x in present_task_metrics])
        avg_pr = _mean([x[1] for x in present_task_metrics])
        avg_f1 = _mean([x[2] for x in present_task_metrics])

        first_rr_crr = _mean(first_recalls_crr)
        first_pr_crr = _mean(first_precs_crr)
        first_f1_crr = (
            2 * first_rr_crr * first_pr_crr / (first_rr_crr + first_pr_crr)
            if (first_rr_crr + first_pr_crr) > 0 else 0.0
        )

        result = {
            'CRR': {'recall_at_2s': rr_crr, 'precision_at_2s': pr_crr, 'f1_at_2s': f1_crr, 'n': len(recalls_crr)},
            'REC': {'recall_at_2s': rr_rec, 'precision_at_2s': pr_rec, 'f1_at_2s': f1_rec, 'n': len(recalls_rec)},
            'SSR': {'recall_at_2s': rr_ssr, 'precision_at_2s': pr_ssr, 'f1_at_2s': f1_ssr, 'n': len(recalls_ssr)},
            'average_recall_at_2s': avg_rr,
            'average_precision_at_2s': avg_pr,
            'average_f1_at_2s': avg_f1,
            'CRR_first_response': {
                'recall_at_2s': first_rr_crr,
                'precision_at_2s': first_pr_crr,
                'f1_at_2s': first_f1_crr,
                'n': len(first_recalls_crr),
                'micro_precision_at_2s': crr_first_hits / crr_answered if crr_answered else 0.0,
            },
            'CRR_diagnostics': {
                'response_count_samples': crr_response_counts,
                'first_response_timing_samples': crr_first_timing,
                'raw_micro_precision_at_2s': crr_raw_hits / crr_total_responses if crr_total_responses else 0.0,
                'total_valid_responses': crr_total_responses,
            },
        }
        print(
            f'OVO_Timing @2s — avg recall: {avg_rr:.4f}, avg precision: {avg_pr:.4f}, avg F1: {avg_f1:.4f}',
            flush=True,
        )
        dump(result, acc_file)
        return result

def _streaming_system_prompt():
    return STREAMING_SYS_CONSERVATIVE_THREE_PROTOCOL

OVOTiming.effective_question = staticmethod(lambda line: str(line["question"]))
