"""Document loading, chunking and retrieval for grounded answers (RAG) and semantic search.

Plane pushes content (work items, Hub docs) with the project it belongs to. A search always comes with the
projects the caller may read, so a person never retrieves content they could not open in Plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.db.tables import DocumentChunk
from app.llm.generate import Generator
from app.llm.tokens import count_tokens, truncate_to_tokens
from app.search.chunking import split_text
from app.search.ranking import BM25, cosine, hybrid

# Most chunks scored per search: keeps a search fast on a small machine.
MAX_CANDIDATES = 5000


@dataclass
class SourceDocument:
    source_type: str
    source_id: str
    title: str
    text: str
    project_id: str | None = None
    url: str = ""


@dataclass
class Hit:
    source_type: str
    source_id: str
    title: str
    url: str
    project_id: str | None
    text: str
    score: float


async def upsert_documents(
    session: Session, generator: Generator | None, workspace_id: str, documents: list[SourceDocument]
) -> int:
    """Replace the chunks of each document. Returns the number of chunks stored."""
    stored = 0
    for doc in documents:
        session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.workspace_id == workspace_id,
                DocumentChunk.source_type == doc.source_type,
                DocumentChunk.source_id == doc.source_id,
            )
        )
        body = f"{doc.title}\n\n{doc.text}".strip()
        chunks = split_text(body)
        vectors = await generator.embed(chunks) if generator is not None and chunks else None
        for index, chunk in enumerate(chunks):
            session.add(
                DocumentChunk(
                    workspace_id=workspace_id,
                    project_id=doc.project_id,
                    source_type=doc.source_type,
                    source_id=doc.source_id,
                    title=doc.title[:300],
                    url=doc.url[:500],
                    chunk_index=index,
                    text=chunk,
                    embedding=vectors[index] if vectors and index < len(vectors) else None,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            stored += 1
    session.commit()
    return stored


def delete_documents(session: Session, workspace_id: str, source_type: str, source_ids: list[str]) -> None:
    session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.workspace_id == workspace_id,
            DocumentChunk.source_type == source_type,
            DocumentChunk.source_id.in_(source_ids),
        )
    )
    session.commit()


async def search(
    session: Session,
    generator: Generator | None,
    workspace_id: str,
    query: str,
    project_ids: list[str],
    limit: int = 8,
    min_score: float = 0.05,
) -> list[Hit]:
    """Chunks matching `query` in the projects the caller may read (plus workspace-wide content)."""
    if not query.strip():
        return []
    stmt = (
        select(DocumentChunk)
        .where(
            DocumentChunk.workspace_id == workspace_id,
            or_(DocumentChunk.project_id.is_(None), DocumentChunk.project_id.in_(project_ids or ["-"])),
        )
        .order_by(DocumentChunk.updated_at.desc())
        .limit(MAX_CANDIDATES)
    )
    rows = list(session.scalars(stmt))
    if not rows:
        return []
    lexical = BM25([r.text for r in rows]).scores(query)
    semantic = None
    if generator is not None and any(r.embedding for r in rows):
        q = await generator.embed([query])
        if q:
            semantic = [cosine(q[0], r.embedding) if r.embedding else 0.0 for r in rows]
    scores = hybrid(lexical, semantic)
    ranked = sorted(zip(rows, scores), key=lambda p: p[1], reverse=True)
    hits: list[Hit] = []
    for row, score in ranked:
        if score < min_score or len(hits) >= limit:
            break
        hits.append(Hit(row.source_type, row.source_id, row.title, row.url, row.project_id, row.text, round(score, 4)))
    return hits


def context_block(hits: list[Hit], token_budget: int) -> str:
    """Retrieved chunks as numbered sources the model can cite as [1], [2]."""
    if not hits or token_budget <= 0:
        return ""
    lines = ["WORKSPACE CONTENT (cite as [n] when you use it):"]
    used = count_tokens(lines[0])
    for n, hit in enumerate(hits, start=1):
        entry = f'[{n}] {hit.source_type} "{hit.title}": {hit.text}'
        cost = count_tokens(entry)
        if used + cost > token_budget:
            entry = truncate_to_tokens(entry, max(0, token_budget - used))
            if entry:
                lines.append(entry)
            break
        lines.append(entry)
        used += cost
    return "\n".join(lines)
