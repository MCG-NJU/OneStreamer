from __future__ import annotations

import re

from .schema import Sample


POST_PROMPT = "Answer with ONLY the single uppercase letter of the correct option."


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\n", " ").replace("\t", " ")
    text = text.strip().strip('"').strip("'").strip("(").strip(")")
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_single_int(text: str) -> int | None:
    numbers = re.findall(r"\b\d+\b", normalize_text(text))
    return int(numbers[0]) if len(numbers) == 1 else None


def build_question_text(sample: Sample) -> str:
    if sample.eval_mode != "mcq_letter":
        return sample.question
    lines = [f"Question:\n{sample.question}", "Options:"]
    for letter, value in sample.options.items():
        lines.append(f"{letter}. {value}")
    lines.extend([POST_PROMPT, "Best option:("])
    return "\n".join(lines)


def parse_option(prediction: str, valid_letters: str) -> str:
    text = (prediction or "").strip()
    if not text:
        return ""
    letters = re.escape(valid_letters)
    patterns = (
        rf"^\s*\(?\s*([{letters}])\s*\)?\s*$",
        rf"^\s*\(?\s*([{letters}])\s*\)?[.\-:)]",
        rf"(?:answer|option|best option|correct option)\s*(?:is|:)?\s*\(?\s*([{letters}])\s*\)?",
        rf"\b([{letters}])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    matches = []
    for letter in valid_letters:
        if normalized == letter.lower():
            matches.append(letter)
    return matches[-1] if matches else ""


def score_prediction(sample: Sample, prediction: str) -> tuple[str, int]:
    if sample.eval_mode == "substring_match":
        parsed = normalize_text(prediction)
        target = normalize_text(sample.answer)
        return parsed, int(bool(parsed) and bool(target) and target in parsed)
    if sample.eval_mode == "integer_match":
        parsed_int = extract_single_int(prediction)
        target_int = extract_single_int(sample.answer)
        parsed = "" if parsed_int is None else str(parsed_int)
        return parsed, int(parsed_int is not None and target_int is not None and parsed_int == target_int)

    parsed = parse_option(prediction, "".join(sample.options))
    if not parsed:
        normalized = normalize_text(prediction)
        candidates = []
        for letter, option in sample.options.items():
            option_normalized = normalize_text(option)
            if option_normalized and (normalized == option_normalized or option_normalized in normalized):
                candidates.append((len(option_normalized), letter))
        if candidates:
            parsed = sorted(candidates)[-1][1]
    return parsed, int(parsed == sample.answer)
