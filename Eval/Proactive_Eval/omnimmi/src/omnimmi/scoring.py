from __future__ import annotations

import ast

import json

import os

import re

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Any


JUDGE_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness of "
    "generative outputs for question-answer pairs. Compare the predicted answer "
    "with the correct answer and determine whether they match meaningfully. Focus "
    "on semantic correctness and accept valid synonyms or paraphrases."
)

def judge_messages(question: str, answer: str, prediction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {prediction}\n\n"
                "Return only a dictionary with keys 'pred' and 'score'. 'pred' must "
                "be 'yes' or 'no'. 'score' must be a number from 0 to 5, where 5 is "
                "the highest meaningful match. Do not add explanation or a code block."
            ),
        },
    ]

def parse_judge_response(raw: str) -> dict[str, Any]:
    value = (raw or "").strip()
    value = re.sub(r"^```(?:json|python)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    parsed = None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge response is not a dictionary: {value!r}")
    verdict = str(parsed.get("pred", "")).strip().lower()
    if verdict not in ("yes", "no"):
        raise ValueError(f"Invalid judge verdict: {verdict!r}")
    score = float(parsed["score"])
    if not 0.0 <= score <= 5.0:
        raise ValueError(f"Judge score is outside [0, 5]: {score}")
    return {"pred": verdict, "score": score}

def _flatten_predictions(task: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for sample in predictions:
        if task in ("ap", "si"):
            flattened.append(
                {
                    "judge_id": sample["sample_id"],
                    "sample_id": sample["sample_id"],
                    "turn_index": None,
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prediction": sample["prediction"],
                }
            )
        else:
            for turn_index, qa in enumerate(sample["qa"]):
                flattened.append(
                    {
                        "judge_id": f"{sample['sample_id']}:turn-{turn_index:02d}",
                        "sample_id": sample["sample_id"],
                        "turn_index": turn_index,
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "prediction": qa["prediction"],
                    }
                )
    return flattened

def aggregate_main_scores(task: str, judged: list[dict[str, Any]]) -> dict[str, Any]:
    if not judged:
        raise ValueError("No judge results to aggregate")
    if task in ("ap", "si"):
        return {
            "samples": len(judged),
            "accuracy": sum(item["judge"]["pred"] == "yes" for item in judged) / len(judged),
            "average_score_0_to_5": sum(item["judge"]["score"] for item in judged) / len(judged),
        }

    by_sample: dict[str, list[dict[str, Any]]] = {}
    for item in judged:
        by_sample.setdefault(item["sample_id"], []).append(item)
    for turns in by_sample.values():
        turns.sort(key=lambda item: item["turn_index"])

    sample_hits = []
    sample_scores = []
    step_hits: dict[int, list[bool]] = {}
    cumulative_hits: dict[int, list[bool]] = {}
    for turns in by_sample.values():
        hits = [turn["judge"]["pred"] == "yes" for turn in turns]
        scores = [turn["judge"]["score"] for turn in turns]
        sample_hits.append(all(hits))
        sample_scores.append(sum(scores) / len(scores))
        running = True
        for index, hit in enumerate(hits):
            running = running and hit
            step_hits.setdefault(index, []).append(hit)
            cumulative_hits.setdefault(index, []).append(running)

    return {
        "samples": len(by_sample),
        "turns": len(judged),
        "all_turns_sample_accuracy": sum(sample_hits) / len(sample_hits),
        "average_per_sample_score_0_to_5": sum(sample_scores) / len(sample_scores),
        "each_step_accuracy": {
            str(index): sum(values) / len(values) for index, values in sorted(step_hits.items())
        },
        "cumulative_step_accuracy": {
            str(index): sum(values) / len(values)
            for index, values in sorted(cumulative_hits.items())
        },
    }

def pa_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("No PA predictions to score")
    official_first_hit = 0
    macro_precision_sum = 0.0
    macro_iou_sum = 0.0
    total_predictions = 0
    total_in_window = 0
    samples_with_any_hit = 0
    distance_to_interval = []
    first_hit_delays = []

    for sample in predictions:
        gt_start, gt_end = (float(value) for value in sample["answer"])
        response_times = [float(value) for value in sample["response_times_seconds"]]
        hits = [value for value in response_times if gt_start <= value <= gt_end]
        if response_times and gt_start <= response_times[0] <= gt_end:
            official_first_hit += 1
        macro_precision_sum += len(hits) / len(response_times) if response_times else 0.0
        total_predictions += len(response_times)
        total_in_window += len(hits)
        if hits:
            samples_with_any_hit += 1
            first_hit_delays.append(hits[0] - gt_start)
        if response_times:
            first = response_times[0]
            if first < gt_start:
                distance_to_interval.append(gt_start - first)
            elif first > gt_end:
                distance_to_interval.append(first - gt_end)
            else:
                distance_to_interval.append(0.0)
            pred_start, pred_end = response_times[0], response_times[-1]
            intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            macro_iou_sum += intersection / union if union > 0 else 0.0

    count = len(predictions)
    precision = total_in_window / total_predictions if total_predictions else 0.0
    recall = samples_with_any_hit / count
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": count,
        "official_first_trigger_accuracy": official_first_hit / count,
        "official_macro_event_precision": macro_precision_sum / count,
        "official_macro_trigger_span_iou": macro_iou_sum / count,
        "micro_event_precision": precision,
        "sample_recall_any_in_window": recall,
        "precision_recall_f1": f1,
        "mean_first_trigger_distance_to_gt_interval_seconds": (
            sum(distance_to_interval) / len(distance_to_interval) if distance_to_interval else None
        ),
        "mean_first_in_window_delay_from_gt_start_seconds": (
            sum(first_hit_delays) / len(first_hit_delays) if first_hit_delays else None
        ),
        "total_response_events": total_predictions,
        "in_window_response_events": total_in_window,
    }
