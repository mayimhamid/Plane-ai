"""Text splitting: paragraphs first, then sentences, into chunks of about `size` tokens with some overlap.

Overlap keeps a sentence that straddles two chunks findable from either side.
"""

from __future__ import annotations

import re

from app.llm.tokens import count_tokens

_SENTENCE = re.compile(r"(?<=[.!?؟۔])\s+")


def split_text(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= size:
            pieces.append(paragraph)
        else:
            pieces.extend(s.strip() for s in _SENTENCE.split(paragraph) if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        tokens = count_tokens(piece)
        if tokens > size:  # one huge sentence: hard cut by words
            words = piece.split()
            step = max(1, int(len(words) * size / tokens))
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i : i + step]))
            continue
        if current and current_tokens + tokens > size:
            chunks.append("\n".join(current))
            # carry the tail over as overlap
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                t = count_tokens(prev)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            current, current_tokens = tail, tail_tokens
        current.append(piece)
        current_tokens += tokens
    if current:
        chunks.append("\n".join(current))
    return chunks
