"""Tests for orchestration/model-init/init_model.py.

Uses a local HTTP server serving a few KB of dummy bytes: nothing is
downloaded from the internet.
"""

import hashlib
import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "init_model", ROOT / "orchestration" / "model-init" / "init_model.py"
)
init_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(init_model)

PAYLOAD = bytes(range(256)) * 40  # 10 KiB fake "model"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeServer:
    """Serves PAYLOAD at /model.gguf with Range support; records requests."""

    def __init__(self, payload: bytes = PAYLOAD, redirect_to: str | None = None):
        self.payload = payload
        self.redirect_to = redirect_to
        self.force_status: int | None = None
        self.requests: list[dict[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append(dict(self.headers))
                if outer.force_status:
                    self.send_response(outer.force_status)
                    self.end_headers()
                    return
                if outer.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", outer.redirect_to)
                    self.end_headers()
                    return
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(outer.payload):
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                else:
                    self.send_response(200)
                body = outer.payload[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.httpd.server_port}/model.gguf"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def cfg(tmp_path: Path, url: str, sha: str | None = PAYLOAD_SHA, token: str | None = None):
    return init_model.Config(
        storage_dir=tmp_path, filename="model.gguf", url=url, sha256=sha, token=token, retries=1
    )


def test_fresh_download(tmp_path, server):
    c = cfg(tmp_path, server.url())
    assert init_model.ensure_model(c) == c.target
    assert c.target.read_bytes() == PAYLOAD
    assert not c.partial.exists()
    assert c.marker.exists()


def test_second_run_is_noop(tmp_path, server):
    c = cfg(tmp_path, server.url())
    init_model.ensure_model(c)
    init_model.ensure_model(c)
    init_model.ensure_model(c)
    assert len(server.requests) == 1  # no re-download on later startups


def test_resumes_partial_download(tmp_path, server):
    c = cfg(tmp_path, server.url())
    c.partial.write_bytes(PAYLOAD[:4000])  # simulate an interrupted download
    init_model.ensure_model(c)
    assert server.requests[0]["Range"] == "bytes=4000-"
    assert c.target.read_bytes() == PAYLOAD


def test_complete_partial_is_finalized(tmp_path, server):
    c = cfg(tmp_path, server.url())
    c.partial.write_bytes(PAYLOAD)  # crashed after download, before rename
    init_model.ensure_model(c)
    assert c.target.read_bytes() == PAYLOAD


def test_checksum_mismatch_fails_and_cleans_up(tmp_path, server):
    c = cfg(tmp_path, server.url(), sha="0" * 64)
    with pytest.raises(init_model.InitError, match="checksum mismatch"):
        init_model.ensure_model(c)
    assert not c.target.exists()
    assert not c.partial.exists()


def test_corrupt_existing_file_is_quarantined_and_replaced(tmp_path, server):
    c = cfg(tmp_path, server.url())
    c.target.write_bytes(b"garbage")
    init_model.ensure_model(c)
    assert c.target.read_bytes() == PAYLOAD
    assert (tmp_path / "model.gguf.corrupt").read_bytes() == b"garbage"


def test_existing_valid_file_without_marker_is_hashed_not_downloaded(tmp_path, server):
    c = cfg(tmp_path, server.url())
    c.target.write_bytes(PAYLOAD)  # e.g. copied in manually
    init_model.ensure_model(c)
    assert server.requests == []
    assert c.marker.exists()


def test_http_404_is_fatal(tmp_path, server):
    server.force_status = 404
    with pytest.raises(init_model.InitError, match="HTTP 404"):
        init_model.ensure_model(cfg(tmp_path, server.url()))


def test_token_not_forwarded_to_other_host(tmp_path):
    cdn = FakeServer()
    hub = FakeServer(redirect_to=cdn.url(host="localhost"))  # different hostname
    try:
        init_model.ensure_model(cfg(tmp_path, hub.url(), token="secret-token"))
        assert hub.requests[0].get("Authorization") == "Bearer secret-token"
        assert "Authorization" not in cdn.requests[0]
    finally:
        cdn.close()
        hub.close()


@pytest.mark.parametrize(
    "env,match",
    [
        ({"PLANE_AI_MODEL_URL": "http://example.com/m.gguf"}, "https"),
        ({"PLANE_AI_LLM_MODEL_FILE": "../evil.gguf"}, "bare file name"),
        ({"PLANE_AI_MODEL_URL": "https://example.com/m.gguf"}, "SHA256 is required"),
        ({"PLANE_AI_MODEL_SHA256": "abc"}, "64 hex"),
    ],
)
def test_config_rejects_bad_env(env, match):
    with pytest.raises(init_model.InitError, match=match):
        init_model.Config.from_env(env)


def test_config_defaults_use_pinned_model():
    c = init_model.Config.from_env({})
    assert c.filename == "Qwen3.5-2B-Q4_K_M.gguf"
    assert c.sha256 == init_model.DEFAULT_SHA256
    assert c.url.startswith("https://huggingface.co/")
    assert c.storage_dir == Path("/models")
