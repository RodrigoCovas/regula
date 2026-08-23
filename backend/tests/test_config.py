"""Configuration seam: one env var selects demo (default) or live; misconfigs fail fast.

Per spec #9: REGULA_MODE selects the mode before any request is served;
Live mode without OPENROUTER_API_KEY refuses to start with a clear message.
"""

import pytest

from src.config import ConfigurationError, Mode, load_settings


def test_default_mode_is_demo(monkeypatch):
    monkeypatch.delenv("REGULA_MODE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = load_settings()
    assert settings.regula_mode == Mode.demo


def test_live_mode_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    settings = load_settings()
    assert settings.regula_mode == Mode.live
    assert settings.openrouter_api_key is not None


def test_live_mode_without_api_key_raises_configuration_error(monkeypatch):
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert "OPENROUTER_API_KEY" in str(excinfo.value)


def test_invalid_mode_value_raises_configuration_error_not_silent_demo(monkeypatch):
    monkeypatch.setenv("REGULA_MODE", "banana")
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert "REGULA_MODE" in str(excinfo.value)
