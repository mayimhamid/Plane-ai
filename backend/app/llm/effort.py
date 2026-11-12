"""Effort: how much of the machine one AI request may use.

An admin picks it per workspace in Plane (Dashboard > AI). It trades quality and speed against resources:

- ``high``: all configured CPU threads, the full context window, longer answers, and a second pass where a
  reviewer checks the draft (writer/evaluator loop). Fastest per token and best answers, heaviest on the box.
- ``medium``: half the threads, a mid-size context, one pass. The default.
- ``low``: one or two threads, a small context, short answers, no extra passes, and deterministic results
  wherever the numbers already tell the story. Uses the least CPU and memory; each answer takes longer.

Ollama takes threads and context per request, so the profile applies directly. llama.cpp fixes threads at
start-up, so there the profile limits context, answer length and extra passes only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Effort(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class EffortProfile:
    effort: Effort
    threads: int
    context_tokens: int
    max_answer_tokens: int
    temperature: float
    # Writer/evaluator rounds after the first draft (0 = single pass).
    refine_rounds: int
    # Chat history kept verbatim; older turns are folded into a running summary.
    history_messages: int
    # Retrieved chunks given to the model for a grounded answer.
    retrieval_chunks: int
    # Team analysis: ask the model only about people whose numbers are unclear.
    llm_only_when_unclear: bool

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["effort"] = self.effort.value
        return data


def profile_for(effort: Effort | str, max_threads: int, max_context: int) -> EffortProfile:
    """The concrete limits for an effort level on this machine."""
    effort = Effort(effort)
    if effort is Effort.HIGH:
        return EffortProfile(
            effort=effort,
            threads=max_threads,
            context_tokens=max_context,
            max_answer_tokens=1024,
            temperature=0.3,
            refine_rounds=1,
            history_messages=12,
            retrieval_chunks=6,
            llm_only_when_unclear=False,
        )
    if effort is Effort.MEDIUM:
        return EffortProfile(
            effort=effort,
            threads=max(1, max_threads // 2),
            context_tokens=min(max_context, 4096),
            max_answer_tokens=512,
            temperature=0.2,
            refine_rounds=0,
            history_messages=8,
            retrieval_chunks=4,
            llm_only_when_unclear=False,
        )
    return EffortProfile(
        effort=effort,
        threads=min(2, max_threads),
        context_tokens=min(max_context, 2048),
        max_answer_tokens=256,
        temperature=0.1,
        refine_rounds=0,
        history_messages=4,
        retrieval_chunks=2,
        llm_only_when_unclear=True,
    )
