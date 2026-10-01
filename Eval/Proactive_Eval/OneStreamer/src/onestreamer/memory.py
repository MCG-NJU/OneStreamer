from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .prompts import MEMORY_POSTAMBLE, MEMORY_PREAMBLE
from .prompts import get_caption_control_protocol


TAG_PATTERN = re.compile(r"^\s*</([A-Za-z]+)>\s*(.*)$", re.DOTALL)


@dataclass(frozen=True)
class MemoryEvent:
    kind: str
    start: float
    end: float
    text: str
    raw: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "MemoryEvent":
        return cls(
            kind=str(value["kind"]),
            start=float(value["start"]),
            end=float(value["end"]),
            text=str(value["text"]),
            raw=str(value.get("raw", value["text"])),
        )


def parse_caption_output(
    raw: str,
    start: float,
    end: float,
    control_protocol: str | None = None,
) -> MemoryEvent:
    match = TAG_PATTERN.match(raw or "")
    if not match:
        return MemoryEvent("invalid", start, end, (raw or "").strip(), raw or "")
    protocol = get_caption_control_protocol(control_protocol=control_protocol)
    kind = protocol.tag_to_kind.get(match.group(1))
    if kind is None:
        return MemoryEvent("invalid", start, end, (raw or "").strip(), raw or "")
    return MemoryEvent(kind, start, end, match.group(2).strip(), raw)


def _time(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _line(event: MemoryEvent) -> str:
    return f"[{_time(event.start)}s-{_time(event.end)}s] {event.text}"


class MemoryBank:
    MODES = {"none", "hierarchical"}

    def __init__(self, events: Iterable[MemoryEvent] = ()):
        self.events = [event for event in events if event.kind in {"response", "summary"} and event.text]

    @property
    def captions(self) -> list[MemoryEvent]:
        return [event for event in self.events if event.kind == "response"]

    @property
    def summaries(self) -> list[MemoryEvent]:
        return [event for event in self.events if event.kind == "summary"]

    def render(self, mode: str, events: Iterable[MemoryEvent] | None = None) -> str:
        if mode not in self.MODES:
            raise ValueError(f"Unknown memory mode {mode!r}; choices: {sorted(self.MODES)}")
        if mode == "none":
            return ""
        selected = list(self.events if events is None else events)
        captions = [event for event in selected if event.kind == "response"]
        summaries = [event for event in selected if event.kind == "summary"]
        lines = [MEMORY_PREAMBLE]
        lines.extend(["<SEMANTIC_SUMMARIES>", *map(_line, summaries), "</SEMANTIC_SUMMARIES>"])
        lines.extend(["<EVENT_CAPTIONS>", *map(_line, captions), "</EVENT_CAPTIONS>"])
        lines.append(MEMORY_POSTAMBLE)
        return "\n".join(lines)

    def render_with_budget(self, mode: str, question: str, tokenizer, max_tokens: int) -> tuple[str, dict]:
        text = self.render(mode)
        if not text:
            return "", {"memory_tokens": 0, "memory_events": 0, "truncated": False}
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count <= max_tokens:
            return text, {
                "memory_tokens": token_count,
                "memory_events": len(self.events),
                "truncated": False,
            }

        candidates = list(reversed(self.summaries)) + self._rank_captions(question)

        selected: list[MemoryEvent] = []
        for event in candidates:
            if event in selected:
                continue
            proposal = selected + [event]
            proposal_text = self.render(mode, sorted(proposal, key=lambda item: (item.end, item.kind)))
            if len(tokenizer.encode(proposal_text, add_special_tokens=False)) <= max_tokens:
                selected = proposal
        selected.sort(key=lambda item: (item.end, item.kind))
        text = self.render(mode, selected)
        return text, {
            "memory_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "memory_events": len(selected),
            "memory_events_total": len(self.events),
            "truncated": True,
        }

    def _rank_captions(self, question: str) -> list[MemoryEvent]:
        terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        stop = {"a", "an", "the", "is", "are", "was", "were", "what", "where", "who", "how", "did", "do", "to", "in", "on", "of", "and", "i"}
        terms -= stop
        captions = self.captions
        total = max(1, len(captions) - 1)
        scored = []
        for index, event in enumerate(captions):
            event_terms = set(re.findall(r"[a-z0-9]+", event.text.lower()))
            overlap = len(terms & event_terms)
            recency = index / total
            anchor = 1.0 if index in {0, len(captions) - 1} or index % max(1, len(captions) // 8) == 0 else 0.0
            scored.append((4.0 * overlap + recency + anchor, index, event))
        return [event for _, _, event in sorted(scored, reverse=True)]
