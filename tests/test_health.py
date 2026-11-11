import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.core.config import Settings
from app.llm.runtime import RuntimeStatus
from app.main import create_app


class FakeDatabase:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.disposed = False

    def ping(self) -> None:
        if not self.healthy:
            raise ConnectionError("db down")

    def dispose(self) -> None:
        self.disposed = True


class FakeRuntime:
    def __init__(self, status: RuntimeStatus = RuntimeStatus.OK) -> None:
        self.status = status
        self.closed = False

    async def check(self) -> RuntimeStatus:
        return self.status

    async def aclose(self) -> None:
        self.closed = True


def make_client(db: FakeDatabase, llm: FakeRuntime | None = None, **settings_kw) -> TestClient:
    settings = Settings(_env_file=None, **settings_kw)
    llm = llm or FakeRuntime()
    return TestClient(
        create_app(settings, database_factory=lambda _s: db, runtime_client_factory=lambda _s: llm)
    )


@pytest.fixture
def db() -> FakeDatabase:
    return FakeDatabase()


def test_health_ok(db):
    with make_client(db) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "service": "plane-ai",
        "version": __version__,
        "environment": "development",
    }


def test_health_ok_even_when_db_down():
    with make_client(FakeDatabase(healthy=False)) as client:
        assert client.get("/health").status_code == 200


def test_ready_ok(db):
    with make_client(db) as client:
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ready",
        "checks": {"database": "ok", "llm_runtime": "ok"},
        "llm": {"runtime": "ollama", "model": "qwen3.5-2b"},
    }


def test_ready_503_when_db_down():
    with make_client(FakeDatabase(healthy=False)) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"
    assert resp.json()["checks"] == {"database": "unavailable", "llm_runtime": "ok"}


@pytest.mark.parametrize("llm_status", ["unavailable", "loading", "model_missing"])
def test_ready_503_when_llm_not_ok(db, llm_status):
    with make_client(db, FakeRuntime(RuntimeStatus(llm_status))) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["checks"] == {"database": "ok", "llm_runtime": llm_status}


def test_shutdown_closes_database_and_runtime_client(db):
    llm = FakeRuntime()
    with make_client(db, llm):
        assert not db.disposed and not llm.closed
    assert db.disposed and llm.closed


def test_docs_disabled_in_production(db):
    with make_client(db, app_env="production", db_password="x") as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/health").json()["environment"] == "production"
