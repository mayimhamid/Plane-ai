"""Prompt templates, in one place and versioned.

Every prompt the service sends is built here from a named template. The version is stored with each answer
(conversation messages, analysis results), so a changed prompt can be traced. ``GET /v1/prompts`` lists them.

Templates use ``string.Template`` (``$name``): user text containing braces can't break them.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str
    description: str

    def render(self, **values: object) -> list[dict[str, str]]:
        clean = {k: "" if v is None else str(v) for k, v in values.items()}
        return [
            {"role": "system", "content": Template(self.system).safe_substitute(clean).strip()},
            {"role": "user", "content": Template(self.user).safe_substitute(clean).strip()},
        ]


_BASE = (
    "You are the assistant inside Plane, a project management tool. Be brief, concrete and factual. "
    "Only use the information you are given; if something is not in it, say you don't know. "
    "Answer in the language of the user's text."
)

PROMPTS: dict[str, PromptTemplate] = {}


def _register(template: PromptTemplate) -> PromptTemplate:
    PROMPTS[template.name] = template
    return template


CHAT = _register(
    PromptTemplate(
        name="chat",
        version=1,
        description="Workspace assistant chat, grounded on retrieved workspace content.",
        system=_BASE
        + "\n\nYou help a team member with their work: summarising, planning, writing and finding things."
        "\n$memory\n$context",
        user="$message",
    )
)

SUMMARIZE = _register(
    PromptTemplate(
        name="summarize",
        version=1,
        description="Summary of a work item, report, document or conversation.",
        system=_BASE
        + " Write a summary for a busy teammate: what it is, its current state, what is done, what is left, "
        "and any risk or blocker. Use at most $bullets short bullet points. No introduction.",
        user="Kind: $kind\nTitle: $title\n\n$text",
    )
)

EVALUATE = _register(
    PromptTemplate(
        name="evaluate",
        version=1,
        description="Evaluator agent: checks a draft against its source and lists concrete fixes.",
        system="You review a draft written from a source text. Check only: facts not in the source, important "
        "points of the source that are missing, and wordiness. Score 1-10 (10 = nothing to fix).",
        user="SOURCE:\n$source\n\nDRAFT:\n$draft",
    )
)

REWRITE = _register(
    PromptTemplate(
        name="rewrite",
        version=1,
        description="Writer agent: improves its draft using the evaluator's notes.",
        system=_BASE + " Rewrite the draft so it fixes every note. Keep the same format. Output only the new draft.",
        user="SOURCE:\n$source\n\nDRAFT:\n$draft\n\nNOTES:\n$notes",
    )
)

EXTRACT_TASKS = _register(
    PromptTemplate(
        name="extract_tasks",
        version=1,
        description="Turns a description or a list into task proposals (JSON) for Plane's import preview.",
        system="You turn work descriptions into tasks for Plane. Follow the instructions in the user's message "
        'exactly and answer with JSON only: {"tasks": [ ...task objects... ]}. Each task has at least a short, '
        "actionable title. Never invent people, modules, states or labels that are not listed.",
        user="$prompt",
    )
)

CLASSIFY = _register(
    PromptTemplate(
        name="classify",
        version=1,
        description="Zero-shot document classification into given labels.",
        system="Classify the text into exactly one of these labels: $labels. Give a confidence from 0 to 1.",
        user="$text",
    )
)

SENTIMENT = _register(
    PromptTemplate(
        name="sentiment",
        version=1,
        description="Sentiment of a text (positive, neutral, negative) with the phrases that show it.",
        system="Decide the sentiment of the text: positive, neutral or negative. Quote up to 3 short phrases "
        "that show it. Give a confidence from 0 to 1.",
        user="$text",
    )
)

NLI = _register(
    PromptTemplate(
        name="nli",
        version=1,
        description="Natural language inference: does the evidence support the claim?",
        system="You check claims against evidence. entailment = the evidence supports the claim; neutral = the "
        "evidence neither supports nor contradicts it; contradiction = the evidence goes against it. Judge only "
        "from the evidence given. Give a confidence from 0 to 1 and one sentence of reasoning.",
        user="CLAIM:\n$claim\n\nEVIDENCE:\n$evidence",
    )
)

MEMBER_REVIEW = _register(
    PromptTemplate(
        name="member_review",
        version=1,
        description="Worker agent of the team analysis: reads one person's work evidence.",
        system="You help a manager understand one team member's reported work over a period. You get numbers "
        "and evidence computed by the system: what the person reported (hours, tasks, tasks marked done) and "
        "what the system recorded (activity on work items, documents, state changes). Say whether the reports "
        "are well supported, partly supported or weakly supported by the evidence, and why, citing the numbers. "
        "Be fair: missing evidence can have honest reasons (meetings, offline work, research). Never insult. "
        "Suggest one or two questions the manager could ask. Answer in English.",
        user="Person: $name\nPeriod: $period\nEvidence score: $score/100\n\nFACTS:\n$facts\n\nFLAGS:\n$flags",
    )
)

TEAM_REVIEW = _register(
    PromptTemplate(
        name="team_review",
        version=1,
        description="Supervisor agent of the team analysis: combines the per-person reviews.",
        system="You are the supervisor of a team analysis. Combine the per-person reviews into a short summary "
        "for the manager: the overall picture, who needs a conversation and why, and patterns across the team "
        "(e.g. many tasks marked done without any recorded activity). At most 8 bullet points. Answer in English.",
        user="Period: $period\nTeam numbers: $team_facts\n\nPER-PERSON REVIEWS:\n$reviews",
    )
)

MEMORY = _register(
    PromptTemplate(
        name="memory",
        version=1,
        description="Folds older chat turns into a running summary (conversation memory).",
        system="Update the running summary of a conversation with the new turns. Keep facts, decisions, names, "
        "numbers and open questions. At most 120 words. Output only the summary.",
        user="CURRENT SUMMARY:\n$summary\n\nNEW TURNS:\n$turns",
    )
)


def catalog() -> list[dict[str, object]]:
    return [
        {"name": p.name, "version": p.version, "description": p.description} for p in sorted(PROMPTS.values(), key=lambda p: p.name)
    ]
