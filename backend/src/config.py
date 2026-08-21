"""Regula configuration.

The OpenRouter model is pinned by locked decision (issue #1): the demo uses
deepseek/deepseek-v4-flash:free only, with no environment-variable override
to a paid model. Only the API key comes from the environment.
"""

import os

OPENROUTER_MODEL = "deepseek/deepseek-v4-flash:free"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
