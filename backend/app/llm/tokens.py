"""Token counting and limits without loading a tokenizer.

A real tokenizer would cost memory and start-up time on small machines. For budgeting, a conservative
estimate is enough: about 3.5 characters per token for Latin text, about 1 token per character for CJK and
roughly 2 characters per token for Persian/Arabic. Exact counts come back from the runtime after each call
and are stored with the message.
"""

from __future__ import annotations

import math
import re

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

# Fixed overhead of the chat template per message (role markers etc.).
MESSAGE_OVERHEAD = 4


def count_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    arabic = len(_ARABIC.findall(text))
    rest = len(text) - cjk - arabic
    return cjk + math.ceil(arabic / 2) + math.ceil(rest / 3.5)


def count_messages(messages: list[dict[str, str]]) -> int:
    return sum(count_tokens(m.get("content", "")) + MESSAGE_OVERHEAD for m in messages)


def truncate_to_tokens(text: str, limit: int) -> str:
    """Cut `text` so it fits in about `limit` tokens, keeping the start and marking the cut."""
    if limit <= 0:
        return ""
    if count_tokens(text) <= limit:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid]) <= limit - 3:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + " [...]"


def input_budget(context_tokens: int, max_answer_tokens: int) -> int:
    """Tokens left for the prompt once the answer has its room."""
    return max(256, context_tokens - max_answer_tokens - 64)
