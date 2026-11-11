import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep the developer's real PLANE_AI_* variables out of tests."""
    for key in list(os.environ):
        if key.upper().startswith("PLANE_AI_"):
            monkeypatch.delenv(key)
