"""Configuration: one env var selects Demo mode (default) or Live mode.

Misconfiguration must fail fast at startup, never surface as a mid-request
server error. Live mode requires OPENROUTER_API_KEY and refuses to start
without it; nothing ever silently degrades.
"""

from enum import Enum
from typing import Optional

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings


class Mode(str, Enum):
    demo = "demo"
    live = "live"


class ConfigurationError(RuntimeError):
    """Invalid configuration; startup must refuse to serve rather than degrade."""


class Settings(BaseSettings):
    regula_mode: Mode = Mode.demo
    openrouter_api_key: Optional[SecretStr] = None


def load_settings() -> Settings:
    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration (check REGULA_MODE): {exc}") from exc
    if settings.regula_mode == Mode.live and not settings.openrouter_api_key:
        raise ConfigurationError(
            "REGULA_MODE=live requires OPENROUTER_API_KEY to be set. "
            "Set the key or start with REGULA_MODE=demo (the default)."
        )
    return settings
