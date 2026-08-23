import sys
from pathlib import Path

import pytest

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from src.models import Chunk


@pytest.fixture()
def assert_exactly_one_provision_target():
    """Every retrieved Chunk targets exactly one provision kind — the
    invariant deterministic Citation derivation relies on."""

    def _assert(chunk: Chunk) -> None:
        targets = (chunk.article_number, chunk.recital_number, chunk.annex_number)
        assert sum(value is not None for value in targets) == 1

    return _assert


@pytest.fixture(autouse=True)
def _restore_main_settings():
    """Boot tests may re-load src.main.settings from env; restore it afterwards."""
    import src.main as main

    saved = main.settings
    yield
    main.settings = saved
