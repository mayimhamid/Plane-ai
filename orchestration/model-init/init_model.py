"""Plane AI model initializer.

Makes sure exactly one verified GGUF model file exists in the model storage
directory, then exits. Safe to run on every ``docker compose up``:

* Already present and verified       -> exits in milliseconds (no re-hash, no network).
* Present but never verified          -> hashes once, records the result.
* Present but wrong checksum          -> renamed to ``*.corrupt`` and downloaded again.
* Missing                             -> downloads to ``<file>.part``; resumes an
                                         interrupted ``.part`` with an HTTP Range request.
* Download finished                   -> SHA256 checked, fsync'd, then atomically renamed
                                         to the final name. A half-written file is never
                                         visible under the final name.

Concurrent runs are serialized with a lock file. Standard library only.

Configuration (environment variables):
  PLANE_AI_MODEL_STORAGE_PATH      directory for model files (default /models)
  PLANE_AI_LLM_MODEL_FILE          file name (default Qwen3.5-2B-Q4_K_M.gguf)
  PLANE_AI_MODEL_URL               download URL
  PLANE_AI_MODEL_SHA256            expected SHA256 (hex). Required unless
  PLANE_AI_MODEL_ALLOW_UNVERIFIED  is "true" (not recommended).
  PLANE_AI_HF_TOKEN                optional bearer token (gated Hugging Face repos);
                                   sent only to the URL's own host.
  PLANE_AI_MODEL_DOWNLOAD_RETRIES  attempts before giving up (default 5)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/"
    "f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_K_M.gguf"
)
DEFAULT_SHA256 = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"

CHUNK = 1024 * 1024  # 1 MiB
DISK_HEADROOM = 256 * 1024 * 1024
USER_AGENT = "plane-ai-model-init/1.0"

log = logging.getLogger("model-init")


class InitError(Exception):
    """Fatal, user-actionable error."""


@dataclass(frozen=True)
class Config:
    storage_dir: Path
    filename: str
    url: str
    sha256: str | None
    token: str | None
    retries: int

    @property
    def target(self) -> Path:
        return self.storage_dir / self.filename

    @property
    def partial(self) -> Path:
        return self.storage_dir / f"{self.filename}.part"

    @property
    def marker(self) -> Path:
        return self.storage_dir / f".{self.filename}.verified.json"

    @property
    def lockfile(self) -> Path:
        return self.storage_dir / f".{self.filename}.lock"

    @classmethod
    def from_env(cls, env: dict[str, str] | os._Environ = os.environ) -> "Config":
        filename = env.get("PLANE_AI_LLM_MODEL_FILE", "").strip() or DEFAULT_FILE
        if "/" in filename or "\\" in filename or filename in {".", ".."}:
            raise InitError(f"PLANE_AI_LLM_MODEL_FILE must be a bare file name, got {filename!r}")

        url = env.get("PLANE_AI_MODEL_URL", "").strip() or DEFAULT_URL
        if urllib.parse.urlsplit(url).scheme != "https":
            raise InitError("PLANE_AI_MODEL_URL must be an https:// URL")

        sha = env.get("PLANE_AI_MODEL_SHA256", "").strip().lower()
        if not sha and filename == DEFAULT_FILE and url == DEFAULT_URL:
            sha = DEFAULT_SHA256
        allow_unverified = env.get("PLANE_AI_MODEL_ALLOW_UNVERIFIED", "").lower() == "true"
        if sha:
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise InitError("PLANE_AI_MODEL_SHA256 must be 64 hex characters")
        elif not allow_unverified:
            raise InitError(
                "PLANE_AI_MODEL_SHA256 is required for a custom model "
                "(or set PLANE_AI_MODEL_ALLOW_UNVERIFIED=true)"
            )

        try:
            retries = max(1, int(env.get("PLANE_AI_MODEL_DOWNLOAD_RETRIES", "5")))
        except ValueError as exc:
            raise InitError("PLANE_AI_MODEL_DOWNLOAD_RETRIES must be an integer") from exc

        return cls(
            storage_dir=Path(env.get("PLANE_AI_MODEL_STORAGE_PATH", "").strip() or "/models"),
            filename=filename,
            url=url,
            sha256=sha or None,
            token=env.get("PLANE_AI_HF_TOKEN", "").strip() or None,
            retries=retries,
        )


# --------------------------------------------------------------------------- helpers


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock so two initializers never write the same file."""
    try:
        import fcntl
    except ImportError:  # Windows (local runs): best effort, no lock
        yield
        return
    with open(path, "a+") as fh:
        log.debug("waiting for lock %s", path)
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def sha256_of(path: Path, hasher: "hashlib._Hash | None" = None) -> "hashlib._Hash":
    h = hasher or hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h


def fingerprint(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def marker_matches(cfg: Config) -> bool:
    try:
        data = json.loads(cfg.marker.read_text())
    except (OSError, ValueError):
        return False
    return (
        data.get("file") == cfg.filename
        and (cfg.sha256 is None or data.get("sha256") == cfg.sha256)
        and data.get("fingerprint") == fingerprint(cfg.target)
    )


def write_marker(cfg: Config, digest: str) -> None:
    tmp = cfg.marker.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"file": cfg.filename, "sha256": digest, "fingerprint": fingerprint(cfg.target)})
    )
    os.replace(tmp, cfg.marker)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


class _SameHostAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when a redirect leaves the original host
    (e.g. Hugging Face -> CDN), so the token is never sent elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(
            req.full_url
        ).hostname:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_SameHostAuthRedirect())


# --------------------------------------------------------------------------- core


def verify_existing(cfg: Config) -> bool:
    """True if the target file exists and is verified. Quarantines bad files."""
    if not cfg.target.is_file():
        return False
    if marker_matches(cfg):
        log.info("model present and verified: %s (%s)", cfg.target, human(cfg.target.stat().st_size))
        return True

    log.info("model present, verifying checksum (one-time): %s", cfg.target)
    digest = sha256_of(cfg.target).hexdigest()
    if cfg.sha256 is None or digest == cfg.sha256:
        write_marker(cfg, digest)
        log.info("checksum OK: %s", digest)
        return True

    quarantine = cfg.target.with_name(f"{cfg.filename}.corrupt")
    os.replace(cfg.target, quarantine)
    log.warning(
        "checksum mismatch (expected %s, got %s); moved to %s and re-downloading",
        cfg.sha256,
        digest,
        quarantine,
    )
    return False


def download_once(cfg: Config) -> None:
    """One attempt: download (or resume) into the .part file."""
    offset = cfg.partial.stat().st_size if cfg.partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if cfg.token:
        headers["Authorization"] = f"Bearer {cfg.token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"

    req = urllib.request.Request(cfg.url, headers=headers)
    try:
        resp = _opener.open(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            log.info("server reports .part already complete (%s)", human(offset))
            return
        if exc.code in (401, 403, 404):
            raise InitError(f"download refused: HTTP {exc.code} for {cfg.url}") from exc
        raise

    with resp:
        if offset and resp.status != 206:
            log.info("server does not support resume; restarting download")
            offset = 0
        length = resp.headers.get("Content-Length")
        remaining = int(length) if length and length.isdigit() else None
        total = offset + remaining if remaining is not None else None

        if remaining is not None:
            free = shutil.disk_usage(cfg.storage_dir).free
            if free < remaining + DISK_HEADROOM:
                raise InitError(
                    f"not enough disk space in {cfg.storage_dir}: need {human(remaining)}, "
                    f"free {human(free)}"
                )

        log.info(
            "%s %s -> %s%s",
            "resuming" if offset else "downloading",
            cfg.filename,
            cfg.partial,
            f" ({human(total)})" if total else "",
        )
        mode = "ab" if offset else "wb"
        done, last_log = offset, time.monotonic()
        with open(cfg.partial, mode) as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                if time.monotonic() - last_log >= 10:
                    pct = f" {done * 100 / total:5.1f}%" if total else ""
                    log.info("progress:%s %s", pct, human(done))
                    last_log = time.monotonic()
            out.flush()
            os.fsync(out.fileno())

        if total is not None and done != total:
            raise ConnectionError(f"incomplete download: {done} of {total} bytes")


def finalize(cfg: Config) -> None:
    """Verify the .part file and atomically move it into place."""
    digest = sha256_of(cfg.partial).hexdigest()
    if cfg.sha256 is not None and digest != cfg.sha256:
        cfg.partial.unlink(missing_ok=True)
        raise InitError(
            f"checksum mismatch after download (expected {cfg.sha256}, got {digest}); "
            "partial file deleted"
        )
    os.replace(cfg.partial, cfg.target)
    with contextlib.suppress(OSError):  # make the rename durable
        dir_fd = os.open(cfg.storage_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    write_marker(cfg, digest)
    log.info("model ready: %s (sha256 %s)", cfg.target, digest)


def ensure_model(cfg: Config) -> Path:
    cfg.storage_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(cfg.storage_dir, os.W_OK):
        uid = getattr(os, "getuid", lambda: "?")()
        raise InitError(f"model storage {cfg.storage_dir} is not writable by uid {uid}")

    with file_lock(cfg.lockfile):
        if verify_existing(cfg):
            return cfg.target
        if cfg.sha256 is None:
            log.warning("no checksum configured: the download will NOT be verified")

        for attempt in range(1, cfg.retries + 1):
            try:
                download_once(cfg)
                finalize(cfg)
                return cfg.target
            except InitError:
                raise
            except (OSError, urllib.error.URLError, ConnectionError) as exc:
                if attempt == cfg.retries:
                    raise InitError(f"download failed after {attempt} attempts: {exc}") from exc
                delay = min(60, 2**attempt)
                log.warning("attempt %d/%d failed (%s); retrying in %ds", attempt, cfg.retries, exc, delay)
                time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("PLANE_AI_LOG_LEVEL", "INFO").upper() or "INFO",
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        ensure_model(Config.from_env())
    except InitError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted; partial download kept for resume")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
