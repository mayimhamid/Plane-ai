"""Background jobs, daily review, Ask and Answer, feedback and the training export."""

from __future__ import annotations

import json
import time

from app.agents.ask import lexical_route, unsupported_numbers
from app.agents.daily import DayRow, person_verdict, rule_check
from app.llm.generate import ThinkFilter, strip_think

from .test_ai import AUTH, FakeGenerator, make_client


def wait(client, job_id, **params):
    query = {"workspace_id": "w", "user_id": "u1", **params}
    for _ in range(200):
        job = client.get(f"/v1/jobs/{job_id}", headers=AUTH, params=query).json()
        if job.get("status") in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def start(client, kind, data, **extra):
    body = {"workspace_id": "w", "user_id": "u1", "kind": kind, "input": data, **extra}
    resp = client.post("/v1/jobs", headers=AUTH, json=body)
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


# ------------------------------------------------------------------ pure parts


def test_think_is_removed_from_answers_and_streams():
    assert strip_think("<think>let me see</think>Answer") == "Answer"
    f = ThinkFilter()
    out = "".join(f.feed(p) for p in ["Hi <thi", "nk>secret", " stuff</th", "ink> there"]) + f.flush()
    assert out == "Hi  there"


def test_rule_check_settles_documented_rows():
    documented = DayRow(id="1", task="Fix bug", minutes=90, note="Fixed the race in the sync worker",
                        evidence={"activity": True, "note": True})
    assert rule_check(documented)[0].value == "plausible"
    research = DayRow(id="2", task="LLM research", minutes=240, note="research")
    assert rule_check(research)[0] is None  # the model has to look


def test_person_verdict_counts_minutes_in_question():
    rows = [
        {"verdict": "inflated", "minutes": 480, "expected_minutes": 60},
        {"verdict": "plausible", "minutes": 60, "expected_minutes": 60},
    ]
    assert person_verdict(rows, 540) == ("concern", 420)
    assert person_verdict([{"verdict": "vague", "minutes": 90, "expected_minutes": None}], 600) == ("check", 90)


def test_number_verification_and_router():
    data = "Ali: 32.5 h reported, 4 reports"
    assert unsupported_numbers("Ali reported 32.5 h in 4 reports", data, "") == []
    assert unsupported_numbers("Ali reported 45 h", data, "") == ["45"]
    assert unsupported_numbers("Ali reported ۴۵ h", data, "") == ["45"]  # Persian digits
    assert lexical_route("who reported the most hours", ["people", "risks", "daily_notes"])[0] == "people"
    assert lexical_route("which tasks are overdue", ["people", "risks"])[0] == "risks"
    # Persian questions find their section too
    assert lexical_route("کی بیشتر از همه تاخیر داشته و کارش برگشت خورده؟", ["people", "quality", "risks"])[0] == "quality"
    assert lexical_route("رضا روی کدام پروژه وقت گذاشته؟", ["people", "projects", "risks"])[0] == "projects"


# ------------------------------------------------------------------ jobs


def test_job_runs_in_background_and_is_private():
    client, _ = make_client(FakeGenerator(["- summary"]))
    with client:
        job_id = start(client, "summarize", {"text": "long text"})
        job = wait(client, job_id)
        assert job["status"] == "done" and job["result"]["summary"] == "- summary"
        other = client.get(f"/v1/jobs/{job_id}", headers=AUTH, params={"workspace_id": "w", "user_id": "u2"})
        assert other.status_code == 404
        wrong_ws = client.get(f"/v1/jobs/{job_id}", headers=AUTH, params={"workspace_id": "x", "user_id": "u1"})
        assert wrong_ws.status_code == 404


def test_manager_jobs_are_shared_with_managers_only():
    client, _ = make_client(FakeGenerator())
    with client:
        job_id = start(client, "daily_review", {"day": "2026-11-12", "people": []}, audience="managers", key="d1")
        wait(client, job_id)
        assert client.get(f"/v1/jobs/{job_id}", headers=AUTH, params={"workspace_id": "w", "user_id": "u2", "manager": True}).status_code == 200
        assert client.get(f"/v1/jobs/{job_id}", headers=AUTH, params={"workspace_id": "w", "user_id": "u2"}).status_code == 404
        latest = client.get("/v1/jobs", headers=AUTH, params={"workspace_id": "w", "user_id": "u2", "manager": True, "kind": "daily_review", "key": "d1"}).json()
        assert latest[0]["id"] == job_id


def test_invalid_input_and_kind_are_rejected_before_queueing():
    client, _ = make_client()
    with client:
        bad = client.post("/v1/jobs", headers=AUTH, json={"workspace_id": "w", "user_id": "u", "kind": "summarize", "input": {}})
        unknown = client.post("/v1/jobs", headers=AUTH, json={"workspace_id": "w", "user_id": "u", "kind": "nope", "input": {}})
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": False, "effort": "low"})
        off = client.post("/v1/jobs", headers=AUTH, json={"workspace_id": "w", "user_id": "u", "kind": "tasks", "input": {"prompt": "x"}})
    assert bad.status_code == 422 and unknown.status_code == 422 and off.status_code == 409


def test_runtime_down_fails_the_job_cleanly():
    client, _ = make_client(FakeGenerator(fail=True))
    with client:
        job = wait(client, start(client, "summarize", {"text": "x"}))
    assert job["status"] == "failed" and job["error"] == "llm_unavailable"


DAY = {
    "day": "2026-11-12",
    "people": [
        {
            "user_id": "p1",
            "name": "Reza",
            "rows": [
                {"id": "r1", "key": "OPS-4", "task": "Change nginx config", "minutes": 480, "note": "nginx config",
                 "evidence": {"activity": False}},
                {"id": "r2", "key": "OPS-5", "task": "Fix login", "minutes": 60, "note": "Fixed the cookie domain for SSO",
                 "evidence": {"activity": True, "note": True}},
            ],
        },
        {
            "user_id": "p2",
            "name": "Sara",
            "rows": [
                {"id": "s1", "task": "Weekly meeting", "kind": "micro", "is_meeting": True, "minutes": 60},
            ],
        },
    ],
}


def test_daily_review_judges_time_against_the_work():
    model = json.dumps(
        {
            "rows": [{"n": 1, "verdict": "inflated", "expected_minutes": 45,
                      "reason": "A routine config change rarely needs a full day.",
                      "question": "What made the change take all day?"}],
            "summary": "Most of the day went to a small config change.",
        }
    )
    client, gen = make_client(FakeGenerator([model]))
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "low"})
        job = wait(client, start(client, "daily_review", DAY, audience="managers"))
    result = job["result"]
    reza = next(p for p in result["people"] if p["user_id"] == "p1")
    rows = {r["id"]: r for r in reza["rows"]}
    assert rows["r1"]["verdict"] == "inflated" and rows["r1"]["source"] == "ai"
    assert rows["r2"]["verdict"] == "plausible" and rows["r2"]["source"] == "rules"
    assert reza["verdict"] == "concern" and reza["questioned_minutes"] == 435
    sara = next(p for p in result["people"] if p["user_id"] == "p2")
    assert sara["verdict"] == "ok"
    assert len(gen.calls) == 1  # low effort: only Reza's doubtful row went to the model
    assert result["people"][0]["user_id"] == "p1"  # concerns first


def test_ask_routes_answers_and_verifies_numbers():
    sections = {
        "people": {"title": "Hours per person", "text": "Reza: 40 h reported, 5 reports\nSara: 38 h reported"},
        "risks": {"title": "Risky tasks", "text": "OPS-9 overdue 4 days"},
    }
    # medium effort: router call, answer with an invented number, then the fix
    gen = FakeGenerator([json.dumps({"sections": ["people"]}), "Reza reported 52 h.", "Reza reported 40 h."])
    client, _ = make_client(gen)
    with client:
        job = wait(client, start(client, "ask", {"question": "Who reported the most hours?", "period": "this week",
                                                   "sections": sections}))
    result = job["result"]
    assert result["answer"] == "Reza reported 40 h." and result["verified"] is True
    assert result["sections"][0] == "people" and result["router"] == "model"
    assert "Risky tasks" not in gen.calls[1]["messages"][0]["content"]  # only the routed section was sent


def test_feedback_becomes_training_data():
    model = json.dumps({"rows": [{"n": 1, "verdict": "inflated", "expected_minutes": 45, "reason": "r"}], "summary": "s"})
    client, _ = make_client(FakeGenerator([model]))
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "low"})
        job_id = wait(client, start(client, "daily_review", DAY, audience="managers"))["id"]
        ok = client.post(f"/v1/jobs/{job_id}/feedback", headers=AUTH,
                         json={"workspace_id": "w", "user_id": "admin", "manager": True, "item": "p1", "rating": -1,
                               "correction": '{"rows": [{"n": 1, "verdict": "plausible"}]}'})
        assert ok.status_code == 201
        export = client.get("/v1/training/export", headers=AUTH, params={"workspace_id": "w"})
    lines = [json.loads(line) for line in export.text.splitlines()]
    assert len(lines) == 1
    example = lines[0]
    assert example["messages"][0]["role"] == "system"
    assert example["messages"][-1] == {"role": "assistant", "content": '{"rows": [{"n": 1, "verdict": "plausible"}]}'}
    assert example["meta"]["corrected"] is True


def test_ask_puts_the_named_person_first_and_gives_them_room():
    rows = "- 2026-11-10 OPS-4 nginx: 480 min; recorded: nothing\n" * 40
    sections = {
        "people": {"title": "Hours", "text": "Reza: 40 h\nSara: 38 h"},
        "quality": {"title": "Quality", "text": "Reza: 3 late"},
        "person": {"title": "About Reza", "text": "# Reza\n" + rows},
    }
    client, gen = make_client(FakeGenerator(["Reza spent 480 min on nginx."]))
    with client:
        client.put("/v1/workspaces/w/settings", headers=AUTH, json={"enabled": True, "effort": "low"})
        job = wait(client, start(client, "ask", {"question": "How did Reza work this week?", "sections": sections}))
    assert job["result"]["sections"][0] == "person"
    prompt = gen.calls[-1]["messages"][0]["content"]
    assert prompt.count("OPS-4 nginx") > 10  # most of the person's rows fit in a small context
