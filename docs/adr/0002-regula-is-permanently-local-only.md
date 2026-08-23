# Regula is permanently local-only

Regula will never be deployed to Render or any other cloud host; it runs locally via Docker Compose. This was made explicit after the deterministic demo shipped: the verified constraint that Render's free tier cannot run Ollama (0.1 CPU / 512 MB, managed Postgres paid-only) removed the original "Local + Render Free Tier" plan, and the project's purpose — a self-contained prototype that stakeholders run on their own machines, with no cloud account required — makes cloud operation a non-goal rather than a deferred decision.

## Consequences

- Local Ollama is a permanent architectural dependency for embeddings; do not abstract toward hosted embedding APIs in anticipation of a cloud deploy that will not happen.
- Stakeholder distribution is this repository plus `docker compose`, not a hosted URL.
- The absence of deployment configuration, CI deploys, and hosting docs is the decision — do not "fix" it.
