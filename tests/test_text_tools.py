"""Token counting, text splitting, BM25 and the cache (pure functions, no app)."""

import json

import pytest

from app.llm.structured import parse_json
from app.llm.tokens import count_tokens, input_budget, truncate_to_tokens
from app.search.chunking import split_text
from app.search.ranking import BM25, cosine, hybrid
from app.services.cache import TTLCache


def test_count_tokens_scripts():
    assert count_tokens("") == 0
    assert 2 <= count_tokens("hello world") <= 4
    assert count_tokens("سلام دنیا") >= 4  # Persian counts denser than Latin


def test_truncate_keeps_limit():
    text = "word " * 1000
    cut = truncate_to_tokens(text, 50)
    assert count_tokens(cut) <= 50 and cut.endswith("[...]")
    assert truncate_to_tokens("short", 50) == "short"


def test_input_budget_leaves_room_for_answer():
    assert input_budget(2048, 256) == 2048 - 256 - 64
    assert input_budget(300, 290) == 256


def test_split_text_overlaps_and_bounds():
    text = "\n\n".join(f"Paragraph {i}. " + "This sentence is filler. " * 30 for i in range(5))
    chunks = split_text(text, size=120, overlap=30)
    assert len(chunks) > 3
    assert all(count_tokens(c) <= 160 for c in chunks)
    assert split_text("") == []


def test_bm25_ranks_relevant_first():
    docs = ["login page broken users cannot sign in", "install anchors in hall b", "weekly report template"]
    scores = BM25(docs).scores("sign in broken")
    assert scores.index(max(scores)) == 0
    assert scores[1] == 0


def test_hybrid_and_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == 0
    assert hybrid([2.0, 1.0], None) == [1.0, 0.5]
    assert hybrid([2.0, 0.0], [0.0, 1.0], weight=0.5) == [0.5, 0.5]


def test_parse_json_tolerates_fences_and_noise():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope it helps') == {"a": 2}
    with pytest.raises(json.JSONDecodeError):
        parse_json("nothing here")


def test_cache_ttl_and_disabled():
    cache = TTLCache(60, max_items=2)
    k = TTLCache.key("a", {"x": 1})
    cache.set(k, 1)
    assert cache.get(k) == 1
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get(k) is None  # evicted (max 2)
    off = TTLCache(0)
    off.set(k, 1)
    assert off.get(k) is None
