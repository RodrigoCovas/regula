# Provider configuration is environment-driven

The LLM was deliberately pinned to OpenRouter's `upstage/solar-pro4` with no env override (a repin from the free nemotron tier, recorded as an amendment on spec #9) — the pin bought eval reproducibility at the cost of flexibility. Interviewers and operators now bring their own key and model, so `LLM_MODEL`, `LLM_BASE_URL`, and the API key are environment configuration, using the OpenAI-compatible base-URL convention (`/chat/completions` appended to the base). Solar Pro 4 stays the default and the README's suggested model; the pin's history (free-tier rate limits, upstream errors) remains visible in the query log.

## Consequences

- Eval results are model-sensitive: the README quotes the model that produced them, and numbers across models are not directly comparable.
- OpenRouter is the tested path; other OpenAI-compatible endpoints are expected to work but untested.
