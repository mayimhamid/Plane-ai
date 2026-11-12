"""Chat with history, memory and grounded answers.

- History: every turn is stored (``ai_message``), so a conversation survives a restart (checkpointer).
- Memory: only the last N turns (by effort) go to the model verbatim; older turns are folded into a running
  summary that is sent instead (conversation summarization).
- Limits: the prompt is cut to the context window minus the room kept for the answer (token counting).
- Grounding: chunks retrieved from the workspace are added as numbered sources (RAG).
- Streaming: the answer is yielded piece by piece and stored when complete.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.tables import Conversation, Message
from app.llm.effort import EffortProfile
from app.llm.generate import GenerationError, Generator, Usage
from app.llm.tokens import count_messages, count_tokens, input_budget, truncate_to_tokens
from app.prompts import CHAT, MEMORY
from app.services.documents import Hit, context_block, search

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 8000


class ConversationNotFound(LookupError):
    pass


@dataclass
class ChatTurn:
    conversation: Conversation
    messages: list[dict[str, str]]
    hits: list[Hit]


def get_conversation(session: Session, workspace_id: str, user_id: str, conversation_id: str) -> Conversation:
    conv = session.get(Conversation, conversation_id)
    # A conversation belongs to one person in one workspace: never readable by anyone else.
    if conv is None or conv.workspace_id != workspace_id or conv.user_id != user_id:
        raise ConversationNotFound(conversation_id)
    return conv


def list_conversations(session: Session, workspace_id: str, user_id: str, limit: int = 30) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .where(Conversation.workspace_id == workspace_id, Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def _stored(conv: Conversation) -> list[Message]:
    return sorted(conv.messages, key=lambda m: m.position)


async def fold_memory(session: Session, generator: Generator, profile: EffortProfile, conv: Conversation) -> None:
    """Fold turns older than the verbatim window into the running summary."""
    stored = _stored(conv)
    keep_from = max(0, len(stored) - profile.history_messages)
    pending = [m for m in stored[conv.summarized_count : keep_from]]
    if not pending:
        return
    turns = "\n".join(f"{m.role}: {m.content}" for m in pending)
    try:
        result = await generator.complete(
            MEMORY.render(summary=conv.summary or "(empty)", turns=truncate_to_tokens(turns, profile.context_tokens // 2)),
            profile,
        )
    except GenerationError as exc:
        logger.warning("memory summary skipped: %s", exc)
        return
    conv.summary = result.text.strip()[:2000]
    conv.summarized_count = keep_from
    session.commit()


async def prepare_turn(
    session: Session,
    generator: Generator,
    profile: EffortProfile,
    *,
    workspace_id: str,
    user_id: str,
    conversation_id: str | None,
    message: str,
    project_ids: list[str],
    use_workspace: bool = True,
) -> ChatTurn:
    message = message.strip()[:MAX_MESSAGE_CHARS]
    if conversation_id:
        conv = get_conversation(session, workspace_id, user_id, conversation_id)
    else:
        conv = Conversation(workspace_id=workspace_id, user_id=user_id, title=message[:80], summary="", summarized_count=0)
        session.add(conv)
        session.flush()

    stored = _stored(conv)
    session.add(Message(conversation_id=conv.id, position=len(stored), role="user", content=message, prompt_tokens=count_tokens(message)))
    session.commit()
    session.refresh(conv)

    await fold_memory(session, generator, profile, conv)

    hits = (
        await search(session, generator, workspace_id, message, project_ids, limit=profile.retrieval_chunks)
        if use_workspace
        else []
    )
    budget = input_budget(profile.context_tokens, profile.max_answer_tokens)
    memory = f"\nEarlier in this conversation: {conv.summary}" if conv.summary else ""
    context = context_block(hits, budget // 2)
    base = CHAT.render(memory=memory, context=context, message=message)
    system = base[0]

    history = [{"role": m.role, "content": m.content} for m in _stored(conv)[conv.summarized_count :]]
    history = history[-profile.history_messages :]
    # Drop the oldest turns until the whole prompt fits (the last one is the new question and always stays).
    while len(history) > 1 and count_messages([system, *history]) > budget:
        history.pop(0)
    if count_messages([system, *history]) > budget:
        history[-1] = {"role": "user", "content": truncate_to_tokens(history[-1]["content"], budget // 2)}
    return ChatTurn(conversation=conv, messages=[system, *history], hits=hits)


def save_answer(session: Session, conv: Conversation, text: str, usage: Usage, effort: str) -> Message:
    position = len(_stored(conv))
    msg = Message(
        conversation_id=conv.id,
        position=position,
        role="assistant",
        content=text,
        prompt_tokens=usage.prompt_tokens,
        answer_tokens=usage.answer_tokens or count_tokens(text),
        prompt_version=f"{CHAT.name}@{CHAT.version}",
        effort=effort,
    )
    session.add(msg)
    session.commit()
    return msg


async def stream_answer(
    session: Session, generator: Generator, profile: EffortProfile, turn: ChatTurn
) -> AsyncIterator[str]:
    """Yield the answer piece by piece; store it when done (also when the client disconnects early)."""
    usage = Usage()
    parts: list[str] = []
    try:
        async for piece in generator.stream(turn.messages, profile, usage):
            parts.append(piece)
            yield piece
    finally:
        text = "".join(parts).strip()
        if text:
            save_answer(session, turn.conversation, text, usage, profile.effort.value)
