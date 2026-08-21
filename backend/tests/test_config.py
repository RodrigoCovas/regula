import importlib

import src.config as config


def test_openrouter_model_pinned_to_free_deepseek():
    assert config.OPENROUTER_MODEL == "deepseek/deepseek-v4-flash:free"


def test_model_not_env_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    importlib.reload(config)
    assert config.OPENROUTER_MODEL == "deepseek/deepseek-v4-flash:free"
