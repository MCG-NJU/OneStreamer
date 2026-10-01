import json,re

PROMPTS = {'vt': '\nDetermine if the GPT text contains intent to conclude the conversation.\nUse a 0-5 scale where 0 indicates no termination signals and 5 shows explicit closure intent.\n5 = Contains explicit closure phrases (e.g., "Goodbye", "End of discussion")\n4 = Implies conclusion with summary + sign-off (e.g., "In summary...feel free to ask more")\n3 = Neutral statement with closing tendency (e.g., "Let me know if needed")\n2 = Ambiguous transitional phrasing (e.g., "Anyway...")\n1 = Contains weak closure cues (e.g., single emoji/period)\n0 = No termination indicators, continues dialog flow\n\nDO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.\nDO NOT INCLUDE ANY MARKDOWN FORMAT.\nOnly provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}\n', 'vw': '\nDetermine if the GPT text expresses greeting intent.\nUse a 0-5 scale where 0 indicates no greeting signals and 5 shows explicit welcoming language.\n5 = Explicit welcoming phrases ("Hello!", "How can I assist?")\n4 = Contextual opening (Matching time-based greetings like "Good morning")\n3 = Service-oriented initiation ("What can I help with?")\n2 = Neutral conversational starters ("Shall we begin?")\n1 = Implicit opening cues (Single welcoming emoji)\n0 = No greeting/dialogue initiation detected\n\nDO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.\nDO NOT INCLUDE ANY MARKDOWN FORMAT.\nOnly provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}\n', 'vi': '\nDetermine if the GPT text contains cessation intent (stopping speaking/concluding).\nUse a 0-5 scale where 0 indicates continued engagement and 5 shows explicit termination.\n5 = Explicit acknowledgment ("I\'ll stop here") + closure\n4 = Implicit cessation recognition ("Unless you need more...")\n3 = Passive-agreement cessation ("As requested, ending now")\n2 = Ambiguous continuation ("...but") after stop cue\n1 = Partial acknowledgment without closure\n0 = Complete continuation disregard\n\nDO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.\nDO NOT INCLUDE ANY MARKDOWN FORMAT.\nOnly provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}\n', 'aw': '\nEvaluate video anomaly response from GPT with the following metric, taking into account the total score of 5 points,\nwith separate scores for Description Consistency between ground truth and Advice Rationality:\n\nDescription Consistency between Ground Truth:\n3 = Core elements match without errors\n2 = Core elements match with minor errors\n1 = Superficial match\n0 = Key element errors or contradictory\n\nAdvice Rationality:\n2 = Actionable & Safe & Logically sound\n1 = Generally appropriate\n0 = Dangerous/hallucinated\n\nDO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.\nDO NOT INCLUDE ANY MARKDOWN FORMAT.\nOnly provide the Total score and reason in JSON format. For example, {"description":3, "advice":2, "reason": "this is a reason for the score"}\n', 'hr': '\nEvaluate alignment between Ground Truth and GPT Text regarding humorous event descriptions.\n5 = Perfect match in humor and delivery\n4 = Preserves main humor, but with minor changes to the story or details\n3 = Only partial humor retention with some deviations\n2 = Only partial humor retention and some important parts are missing\n1 = Superficial similarity only\n0 = No comedic correlation\n\nDO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.\nDO NOT INCLUDE ANY MARKDOWN FORMAT.\nOnly provide the score and reason in JSON format. For example, {"score":3, "reason": "this is a reason for the score"}\n', 'gu': '\nEvaluate gesture response from GPT with the following metric, taking into account the total score of 5 points,\nwith separate scores for gesture recognition and contextual appropriateness of the response:\n\nGesture recognition:\n3 = Precise gesture identification\n2 = Ambiguous gesture reference\n1 = No explicit mention of gestures\n0 = Hallucinated/non-existent gesture\n\nContextual appropriateness:\n2 = Natural integration with dialogue\n1 = Generic but relevant response\n0 = Irrelevant/contradictor response\n\n[Dialogue History] provided for context\n[Gesture] is the ground truth\n[Contextual Reference Text] as a reference, but does not have to match exactly\n\nDO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION.\nDO NOT INCLUDE ANY MARKDOWN FORMAT.\nOnly provide the score and reason in JSON format. For example, {"gesture":3, "context":2, "reason": "this is a reason for the score"}\n'}

def response_is_in_time(response_time, reference, margin=2.0):
    if response_time is None:
        return False
    if isinstance(reference, (list, tuple)):
        return max(0.0, reference[0]) <= response_time <= reference[1] + margin
    return reference <= response_time <= reference + margin

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def parse_json_response(text):
    cleaned = strip_thinking(text)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))

def build_prompt(kind, prediction, ground_truth):
    reference = ground_truth["conversations"][-1]["value"]
    if kind == "gu":
        dialogue = "\n".join(
            f"{turn['from']}: {turn['value']}"
            for turn in ground_truth["conversations"][:2]
        )
        case = (
            f"\n[Dialogue]\n{dialogue}\n"
            f"[Ground-truth gesture]\n{ground_truth['gesture']}\n"
            f"[Reference response]\n{reference}\n"
            f"[Assistant text]\n{prediction}\n"
        )
    elif kind in {"aw", "hr"}:
        case = f"\n[Reference]\n{reference}\n[Assistant text]\n{prediction}\n"
    else:
        case = f"\n[Assistant text]\n{prediction}\n"
    return PROMPTS[kind] + case

def score_payload(kind, payload):
    if kind == "gu":
        score = float(payload["gesture"]) + float(payload["context"])
    elif kind == "aw":
        score = float(payload["description"]) + float(payload["advice"])
    else:
        score = float(payload["score"])
    if not 0 <= score <= 5:
        raise ValueError(f"Judge score outside [0, 5]: {score}")
    return score

def task_summary(records, total):
    time_correct = sum(bool(record.get("time_correct")) for record in records.values())
    successful = [
        record
        for record in records.values()
        if record.get("time_correct") and record.get("judge_status") == "ok"
    ]
    answer_score = sum(float(record["score"]) for record in successful)
    return {
        "total": total,
        "recorded": len(records),
        "time_correct": time_correct,
        "time_accuracy": time_correct / total if total else 0.0,
        "judge_success": len(successful),
        "judge_pending_or_failed": time_correct - len(successful),
        "answer_score": answer_score,
        "answer_average_score_overall": answer_score / total if total else 0.0,
        "answer_average_score_time_correct": (
            answer_score / time_correct if time_correct else 0.0
        ),
        "complete": len(records) == total and len(successful) == time_correct,
    }
