from __future__ import annotations
import json,re,csv
from collections import defaultdict
from pathlib import Path
from typing import Any
JUDGE_INSTRUCTION = (
    "You are an evaluator for a video question answering system. "
    "Your task is to rate whether the predicted answer covers the key points "
    "of the ground truth answer. Use the following scale to assign a score:\n"
    "- 3: Mostly covered; the predicted answer covers all key information in "
    "the ground truth answer, though it may have minor inaccuracies or rephrases.\n"
    "- 2: Partially covered; the predicted answer has some correct information, "
    "but also contains significant inaccuracies or missing key points.\n"
    "- 1: Incorrect; the predicted answer may be related to the ground truth "
    "answer, but most of the information is missing, or the predicted answer is "
    "not relevant to the question or in very poor quality.\n\n"
    "Output the score only, do not add more explanations."
)

def _area_under_line_ratio(
    points: list[list[float]],
    max_x: float,
    max_y: float,
    omega: float = 0.5,
    start_score: float = 0.5,
) -> float:
    if not points:
        return 0.0
    scaled_points = sorted(points, key=lambda point: point[0])
    scaled_points = [(x * (1 - omega), y) for x, y in scaled_points]
    scaled_points.append((max_x, scaled_points[-1][1]))
    prev_y, prev_x, area = start_score, 0.0, 0.0
    max_area = max_x * max_y
    for x1, y1 in scaled_points:
        area += (x1 - prev_x) * prev_y
        prev_x, prev_y = x1, y1
    return area / max_area if max_area > 0 else 0.0

def _stat_metric(
    scored_examples: list[dict[str, Any]],
    omega: float = 0.5,
    max_score: float = 2,
    start_score: float = 0.5,
) -> dict[str, float | int]:
    auc_list: list[float] = []
    for example in scored_examples:
        for turn in example["answer"]:
            judge_scores = turn.get("judge_scores", {})
            if not judge_scores:
                auc_list.append(start_score / max_score)
                continue
            points = [
                [float(timestamp) - turn["reply_timespan"][0], score]
                for timestamp, score in judge_scores.items()
            ]
            max_x = turn["reply_timespan"][1] - turn["reply_timespan"][0]
            auc_list.append(
                _area_under_line_ratio(points, max_x, max_score, omega, start_score)
            )
    if not auc_list:
        return {"mean_auc": 0.0, "num_videos": 0, "num_turns": 0}
    return {
        "mean_auc": sum(auc_list) / len(auc_list),
        "num_videos": len(scored_examples),
        "num_turns": len(auc_list),
    }

def build_inputs(
    rows: list[dict[str, str]],
) -> tuple[
    list[tuple[str, list[dict[str, str]]]],
    dict[str, list[tuple[float, str]]],
    dict[str, dict[str, Any]],
]:
    pred_dict: dict[str, list[tuple[float, str]]] = defaultdict(list)
    gold_dict: dict[str, dict[str, Any]] = {}

    for row in rows:
        qid = str(row["question_id"])
        prediction = json.loads(row["prediction"])
        pred_qid = str(prediction.get("question_id", qid))
        if pred_qid != qid:
            raise ValueError(f"prediction qid mismatch: row={qid} pred={pred_qid}")
        for record in prediction.get("records", []):
            if (
                str(record.get("answerable", "")).lower() == "yes"
                and record.get("model_response")
            ):
                pred_dict[qid].append(
                    (
                        float(record["video_span"][1]),
                        str(record["model_response"]),
                    )
                )

        if qid in gold_dict:
            raise ValueError(f"duplicate question_id in TSV: {qid}")
        gold_dict[qid] = {
            "question_id": qid,
            "conversation": json.loads(row["conversation_json"]),
            "answer": json.loads(row["answer_json"]),
        }

    judge_prompts: list[tuple[str, list[dict[str, str]]]] = []
    for qid, preds in pred_dict.items():
        gold = gold_dict[qid]
        for turn_i, answer in enumerate(gold["answer"]):
            ts0, ts1 = answer["reply_timespan"]
            span_preds = [(t, s) for t, s in preds if ts0 <= t <= ts1]
            added: list[str] = []
            for pred_time, sentence in span_preds:
                if sentence in added:
                    continue
                added.append(sentence)
                user_message = (
                    f"Question: {gold['conversation'][0]['content']}\n"
                    f"Ground Truth Answer: {answer['content']}\n"
                    f"Predicted Answer: {' '.join(added)}"
                )
                messages = [
                    {"role": "system", "content": JUDGE_INSTRUCTION},
                    {"role": "user", "content": user_message},
                ]
                judge_prompts.append(
                    (f"{qid}={turn_i}={pred_time}", messages)
                )

    return judge_prompts, pred_dict, gold_dict

def compute_scores(
    *,
    gold_dict: dict[str, dict[str, Any]],
    pred_dict: dict[str, list[tuple[float, str]]],
    results: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored_data: list[dict[str, Any]] = []
    for qid, gold in gold_dict.items():
        preds = pred_dict.get(qid, [])
        answers = []
        for answer in gold["answer"]:
            ts0, ts1 = answer["reply_timespan"]
            turn = dict(answer)
            turn["preds"] = [
                (t, sentence) for t, sentence in preds if ts0 <= t <= ts1
            ]
            turn["judge_scores"] = {}
            answers.append(turn)
        scored_data.append(
            {
                "question_id": qid,
                "conversation": gold["conversation"],
                "answer": answers,
            }
        )
    scored_by_qid = {entry["question_id"]: entry for entry in scored_data}

    for custom_id, result in results.items():
        match = re.match(r"[123]", str(result.get("content", "")).strip())
        if not match:
            continue
        score = int(match.group()) - 1
        parts = custom_id.rsplit("=", 2)
        if len(parts) != 3:
            continue
        qid, turn_i_text, pred_time_text = parts
        try:
            turn_i = int(turn_i_text)
            pred_time = float(pred_time_text)
        except ValueError:
            continue
        entry = scored_by_qid.get(qid)
        if entry is None or turn_i >= len(entry["answer"]):
            continue
        turn = entry["answer"][turn_i]
        ts0, ts1 = turn["reply_timespan"]
        if ts0 <= pred_time <= ts1:
            turn["judge_scores"][pred_time] = score

    for entry in scored_data:
        for turn in entry["answer"]:
            if len(turn["judge_scores"]) <= 1:
                continue
            related_end = turn.get(
                "related_timespan", turn["reply_timespan"]
            )[1]
            items = sorted(turn["judge_scores"].items())
            last_in_related = 0
            to_delete = []
            for pred_time, score in items:
                if pred_time <= related_end:
                    last_in_related = score
                elif score < last_in_related:
                    to_delete.append(pred_time)
            for pred_time in to_delete:
                del turn["judge_scores"][pred_time]

    metrics: dict[str, Any] = {}
    for omega in (0, 0.5, 1):
        value = _stat_metric(scored_data, omega=omega)
        metrics[f"omega={omega}"] = round(value["mean_auc"] * 100, 2)
    metrics["num_videos"] = len(scored_data)
    metrics["num_turns"] = sum(
        len(entry["answer"]) for entry in scored_data
    )
    return scored_data, metrics
