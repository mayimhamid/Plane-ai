import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.core.config import Settings
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


def make_client(db: FakeDatabase, **settings_kw) -> TestClient:
    settings = Settings(_env_file=None, **settings_kw)
    return TestClient(create_app(settings, database_factory=lambda _s: db))


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
    assert resp.json() == {"status": "ready", "checks": {"database": "ok"}}


def test_ready_503_when_db_down():
    with make_client(FakeDatabase(healthy=False)) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready", "checks": {"database": "unavailable"}}


def test_shutdown_disposes_database(db):
    with make_client(db):
        assert not db.disposed
    assert db.disposed


def test_docs_disabled_in_production(db):
    with make_client(db, app_env="production", db_password="x") as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/health").json()["environment"] == "production"
