from collections import defaultdict

import json

from pathlib import Path

OVO_TASKS = {
    'backward_tracking': {'ASI', 'EPM', 'HLD'},
    'real_time_visual_perception': {'ACR', 'ATR', 'FPD', 'OCR', 'OJR', 'STU'},
    'forward_active_responding': {'CRR', 'REC', 'SSR'},
}

def aggregate(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['model_alias'], row['variant'], row['sample_id'])
        if key in seen:
            raise ValueError(f'Duplicate successful answer: {key}')
        seen.add(key)
        grouped[key[:2]].append(row)
    metrics = []
    for (model, variant), items in sorted(grouped.items()):
        is_ovo = any(r['task'] in OVO_TASKS for r in items)
        buckets = defaultdict(lambda: defaultdict(list))
        for row in items:
            if is_ovo and row['subtask'] not in OVO_TASKS.get(row['task'], set()):
                raise ValueError(f'Unexpected OVO task: {row["task"]}/{row["subtask"]}')
            buckets[row['task']][row['subtask']].append(int(row['hit']))
        subtask = {task: {name: sum(hits)/len(hits) for name, hits in parts.items()}
                   for task, parts in buckets.items()}
        category = {task: sum(parts.values())/len(parts) for task, parts in subtask.items()}
        micro = sum(int(r['hit']) for r in items)/len(items)
        missing = {task: sorted(names-set(subtask.get(task, {})))
                   for task, names in OVO_TASKS.items()} if is_ovo else {}
        complete_categories = not any(missing.values())
        overall = (sum(category.values())/3 if complete_categories else None) if is_ovo else micro
        metrics.append(dict(model=model, variant=variant, n=len(items),
                            overall_accuracy=overall, sample_micro_accuracy=micro,
                            task_accuracy=category, task_subtask_accuracy=subtask,
                            missing_subtasks=missing,
                            aggregation='mean_3_categories_of_mean_subtask_accuracy' if is_ovo else 'sample_micro_accuracy'))
    import numpy as np
    for metric in metrics:
        matching = grouped[(metric['model'], metric['variant'])]
        measured = [r for r in matching if 'input_tokens' in r]
        def distribution(values):
            return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                        p95=float(np.percentile(values, 95)), min=int(min(values)), max=int(max(values)))
        if measured:
            if len(measured) != len(matching):
                raise ValueError('Only some answers contain input token measurements')
            metric['context'] = {field: distribution([r[field] for r in measured])
                                 for field in ['input_tokens', 'visual_tokens', 'text_tokens']}
            metric['context']['memory_tokens'] = distribution([r['memory_meta']['memory_tokens'] for r in measured])
            metric['mean_answer_seconds'] = float(np.mean([r['runtime_seconds'] for r in measured]))
    return dict(rows=len(rows), metrics=metrics)
