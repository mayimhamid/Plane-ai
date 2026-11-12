"""The /v1 AI API against a scripted fake runtime and an in-memory SQLite database."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.models import Base
from app.llm.effort import Effort, profile_for
from app.llm.generate import Generation, GenerationError, Usage, current_trace
from app.llm.runtime import RuntimeStatus
from app.main import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class SqliteDatabase:
    def __init__(self) -> None:
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def ping(self) -> None:
        pass

    def dispose(self) -> None:
        self.engine.dispose()


class FakeRuntime:
    async def check(self) -> RuntimeStatus:
        return RuntimeStatus.OK

    async def aclose(self) -> None:
        pass


class FakeGenerator:
    """Answers from a queue of scripted replies (or a function of the messages); records every call."""

    def __init__(self, replies=None, fail: bool = False) -> None:
        self.replies = list(replies or [])
        self.fail = fail
        self.calls: list[dict] = []

    def _next(self, messages) -> str:
        if self.fail:
            raise GenerationError("down")
        if not self.replies:
            return "ok"
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply

    async def complete(self, messages, profile, json_schema=None) -> Generation:
        self.calls.append({"messages": messages, "profile": profile, "schema": json_schema})
        text = self._next(messages)
        trace = current_trace.get()
        if trace is not None:  # like the real generator: jobs record every call
            trace.append({"messages": messages, "output": text})
        return Generation(text, Usage(10, 5))

    async def stream(self, messages, profile, usage):
        self.calls.append({"messages": messages, "profile": profile, "stream": True})
        text = self._next(messages)
        for word in text.split(" "):
            yield word + " "
        usage.prompt_tokens, usage.answer_tokens = 10, 5

    async def embed(self, texts):
        return None

    async def aclose(self) -> None:
        pass


def make_client(gen: FakeGenerator | None = None, **kw) -> tuple[TestClient, FakeGenerator]:
    gen = gen or FakeGenerator()
    settings = Settings(_env_file=None, service_token=TOKEN, cache_ttl_seconds=0, **kw)
    app = create_app(
        settings,
        database_factory=lambda _s: SqliteDatabase(),
        runtime_client_factory=lambda _s: FakeRuntime(),
        generator_factory=lambda _s: gen,
    )
    return TestClient(app), gen


# ------------------------------------------------------------------ auth and settings


def test_requires_service_token():
    client, _ = make_client()
    with client:
        assert client.get("/v1/status").status_code == 401
        assert client.get("/v1/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/status", headers=AUTH).status_code == 200


def test_status_lists_profiles():
    client, _ = make_client()
    with client:
        data = client.get("/v1/status", headers=AUTH).json()
    assert data["llm"] == "ok"
    assert set(data["profiles"]) == {"high", "medium", "low"}


def test_effort_profiles_trade_resources():
    high, medium, low = (profile_for(e, 8, 8192) for e in (Effort.HIGH, Effort.MEDIUM, Effort.LOW))
    assert high.threads == 8 and medium.threads == 4 and low.threads == 2
    assert high.context_tokens > medium.context_tokens > low.context_tokens
    assert high.max_answer_tokens > low.max_answer_tokens
    assert high.refine_rounds == 1 and low.refine_rounds == 0
    assert low.llm_only_when_unclear and not high.llm_only_when_unclear


def test_settings_default_then_update():
    client, _ = make_client()
    with client:
        got = client.get("/v1/workspaces/ws1/settings", headers=AUTH).json()
        assert got["enabled"] is True and got["effort"] == "medium"
        put = client.put("/v1/workspaces/ws1/settings", headers=AUTH, json={"enabled": True, "effort": "low"}).json()
        assert put["profile"]["threads"] <= 2
        assert client.get("/v1/workspaces/ws1/settings", headers=AUTH).json()["effort"] == "low"


def test_disabled_workspace_gets_409():
    client, _ = make_client()
    with client:
        client.put("/v1/workspaces/ws1/settings", headers=AUTH, json={"enabled": False, "effort": "low"})
        resp = client.post("/v1/summarize", headers=AUTH, json={"workspace_id": "ws1", "text": "hello"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "ai_disabled"


# ------------------------------------------------------------------ summarize (writer/evaluator)


def test_summarize_single_pass_at_medium():
    client, gen = make_client(FakeGenerator(["- done"]))
    with client:
        data = client.post("/v1/summarize", headers=AUTH, json={"workspace_id": "w", "text": "long text"}).json()
    assert data["summary"] == "- done" and data["rounds"] == 0
    assert len(gen.calls) == 1


def test_summarize_refines_at_high_effort():
    gen = FakeGenerator(["draft", json.dumps({"score": 5, "notes": ["add the blocker"]}), "better draft"])
    client, _ = make_client(gen)
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "high"})
        data = client.post("/v1/summarize", headers=AUTH, json={"workspace_id": "w", "text": "src"}).json()
    assert data["summary"] == "better draft"
    assert data["rounds"] == 1 and data["scores"] == [5]


def test_summarize_stops_when_evaluator_is_happy():
    gen = FakeGenerator(["draft", json.dumps({"score": 9, "notes": []})])
    client, _ = make_client(gen)
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "high"})
        data = client.post("/v1/summarize", headers=AUTH, json={"workspace_id": "w", "text": "src"}).json()
    assert data["summary"] == "draft" and data["rounds"] == 0


def test_runtime_down_is_503():
    client, _ = make_client(FakeGenerator(fail=True))
    with client:
        resp = client.post("/v1/summarize", headers=AUTH, json={"workspace_id": "w", "text": "x"})
    assert resp.status_code == 503 and resp.json()["detail"] == "llm_unavailable"


# ------------------------------------------------------------------ structured output


def test_tasks_validated_and_repaired():
    bad = "not json at all"
    good = "```json\n" + json.dumps({"tasks": [{"title": " Install anchors ", "priority": "HIGH", "planned_hours": 3}]}) + "\n```"
    client, gen = make_client(FakeGenerator([bad, good]))
    with client:
        data = client.post("/v1/tasks/propose", headers=AUTH, json={"workspace_id": "w", "prompt": "do things"}).json()
    assert data["tasks"] == [{"title": "Install anchors", "priority": "high", "planned_hours": 3.0}]
    assert gen.calls[0]["schema"]["properties"]["tasks"]  # the schema is sent to the runtime
    assert "not valid" in gen.calls[1]["messages"][-1]["content"]  # repair attempt carries the error


def test_tasks_give_up_after_repair():
    client, _ = make_client(FakeGenerator(["nope", "still nope"]))
    with client:
        resp = client.post("/v1/tasks/propose", headers=AUTH, json={"workspace_id": "w", "prompt": "x"})
    assert resp.status_code == 503


def test_nli_and_classify():
    gen = FakeGenerator(
        [
            json.dumps({"label": "contradiction", "confidence": 0.8, "reasoning": "no activity"}),
            json.dumps({"label": "BUG", "confidence": 0.9}),
        ]
    )
    client, _ = make_client(gen)
    with client:
        nli = client.post(
            "/v1/nli", headers=AUTH, json={"workspace_id": "w", "claim": "I finished it", "evidence": "no changes"}
        ).json()
        cls = client.post(
            "/v1/classify", headers=AUTH, json={"workspace_id": "w", "text": "it crashes", "labels": ["bug", "feature"]}
        ).json()
    assert nli["label"] == "contradiction"
    assert cls["label"] == "bug"  # snapped to a given label


# ------------------------------------------------------------------ search and chat


def _index(client):
    docs = [
        {"source_type": "issue", "source_id": "1", "title": "Login page broken", "text": "Users cannot sign in after the update.", "project_id": "p1"},
        {"source_type": "issue", "source_id": "2", "title": "Install BLE anchors", "text": "Hall B needs 12 anchors.", "project_id": "p2"},
        {"source_type": "doc", "source_id": "3", "title": "Onboarding", "text": "How to sign in the first time.", "project_id": None},
    ]
    return client.post("/v1/documents", headers=AUTH, json={"workspace_id": "w", "documents": docs}).json()


def test_search_respects_projects():
    client, _ = make_client()
    with client:
        assert _index(client)["chunks"] == 3
        only_p2 = client.post(
            "/v1/search", headers=AUTH, json={"workspace_id": "w", "query": "sign in login", "project_ids": ["p2"]}
        ).json()["results"]
        with_p1 = client.post(
            "/v1/search", headers=AUTH, json={"workspace_id": "w", "query": "sign in login", "project_ids": ["p1"]}
        ).json()["results"]
    assert [r["id"] for r in only_p2] == ["3"]  # workspace-wide doc only; p1 is not readable
    assert with_p1[0]["id"] == "1"


def _read_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_streams_stores_and_grounds():
    gen = FakeGenerator(["Hall B needs twelve anchors [1]"])
    client, _ = make_client(gen)
    with client:
        _index(client)
        resp = client.post(
            "/v1/chat",
            headers=AUTH,
            json={"workspace_id": "w", "user_id": "u1", "message": "How many anchors for hall B?", "project_ids": ["p2"]},
        )
        events = _read_sse(resp.text)
        kinds = [e for e, _ in events]
        assert kinds[0] == "meta" and kinds[-1] == "done" and "token" in kinds
        meta = events[0][1]
        assert meta["sources"][0]["id"] == "2"
        conv_id = meta["conversation_id"]
        answer = "".join(d["text"] for e, d in events if e == "token").strip()
        assert answer == "Hall B needs twelve anchors [1]"
        # retrieved content reached the model
        assert "12 anchors" in gen.calls[-1]["messages"][0]["content"]

        stored = client.get(f"/v1/conversations/{conv_id}", headers=AUTH, params={"workspace_id": "w", "user_id": "u1"}).json()
        assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
        # another person can't read it
        other = client.get(f"/v1/conversations/{conv_id}", headers=AUTH, params={"workspace_id": "w", "user_id": "u2"})
        assert other.status_code == 404


def test_chat_memory_folds_old_turns():
    client, gen = make_client(FakeGenerator())
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "low"})  # 4 verbatim
        conv_id = None
        for i in range(4):
            body = {"workspace_id": "w", "user_id": "u", "message": f"question {i}", "use_workspace": False}
            if conv_id:
                body["conversation_id"] = conv_id
            events = _read_sse(client.post("/v1/chat", headers=AUTH, json=body).text)
            conv_id = events[0][1]["conversation_id"]
        stored = client.get(f"/v1/conversations/{conv_id}", headers=AUTH, params={"workspace_id": "w", "user_id": "u"}).json()
    assert len(stored["messages"]) == 8
    assert stored["summary"]  # older turns were folded into the running summary
    last_prompt = gen.calls[-1]["messages"]
    assert "Earlier in this conversation" in last_prompt[0]["content"]
    assert len(last_prompt) - 1 <= 4


# ------------------------------------------------------------------ team analysis (supervisor)


MEMBERS = [
    {"user_id": "a", "name": "Ana", "score": 92, "facts": {"reported_minutes": 2400, "reported_tasks": 6}, "flags": []},
    {"user_id": "b", "name": "Ben", "score": 55, "facts": {"reported_minutes": 2000, "reported_tasks": 5}, "flags": ["3 tasks marked done are still open"]},
    {"user_id": "c", "name": "Cy", "score": 0, "facts": {}, "flags": []},
]


def test_team_analysis_low_effort_asks_llm_only_when_unclear():
    review = json.dumps({"verdict": "partly_supported", "explanation": "Some done tasks are open.", "questions": ["Why?"]})
    client, gen = make_client(FakeGenerator([review]))
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "low"})
        data = client.post(
            "/v1/analyze/team", headers=AUTH, json={"workspace_id": "w", "period": "this week", "members": MEMBERS}
        ).json()
    by = {m["user_id"]: m for m in data["members"]}
    assert by["a"]["source"] == "rules" and by["a"]["verdict"] == "well_supported"
    assert by["b"]["source"] == "ai" and by["b"]["verdict"] == "partly_supported"
    assert by["c"]["verdict"] == "no_data"
    assert len(gen.calls) == 1  # one model call for the one unclear person, none for the team summary
    assert data["summary_source"] == "rules"


def test_team_analysis_high_effort_reviews_everyone_and_summarises():
    review = json.dumps({"verdict": "well_supported", "explanation": "fine", "questions": []})
    summary = json.dumps({"bullets": ["All good"]})
    client, gen = make_client(FakeGenerator([review, review, review, summary]))
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "high"})
        data = client.post(
            "/v1/analyze/team", headers=AUTH, json={"workspace_id": "w", "period": "p", "members": MEMBERS}
        ).json()
    assert data["summary"] == ["All good"] and data["summary_source"] == "ai"
    assert len(gen.calls) == 4


def test_team_analysis_survives_runtime_down():
    client, _ = make_client(FakeGenerator(fail=True))
    with client:
        data = client.post(
            "/v1/analyze/team", headers=AUTH, json={"workspace_id": "w", "period": "p", "members": MEMBERS}
        ).json()
    assert all(m["source"] == "rules" for m in data["members"])
    assert data["summary"]


@pytest.mark.parametrize("path", ["/v1/prompts"])
def test_prompt_catalog(path):
    client, _ = make_client()
    with client:
        names = {p["name"] for p in client.get(path, headers=AUTH).json()}
    assert {"chat", "summarize", "nli", "member_review", "team_review", "memory"} <= names
