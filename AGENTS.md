## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context layout with one root `CONTEXT.md` and ADRs under `docs/adr/`. See `docs/agents/domain.md`.

### Checks

Run after code changes, before committing:

- Backend tests: `python -m pytest backend/tests/ -q`
- Backend type checking: `mypy` (config in `pyproject.toml`, deps in `backend/requirements-dev.txt`)
- Frontend tests: `npm test` (in `frontend/`; Node's built-in runner via `tsx`)
- Frontend type checking: `npm run typecheck` (in `frontend/`)

### LLM API Key

The LLM API key lives in the root `.env` (gitignored) — the one configuration file a Docker deployment (via Compose) and a host-run backend (loaded at settings time) share. App config loads it when tests exercise Live-mode functionality — checking that the file exists is fine, but its contents must never be read, echoed, or surfaced anywhere. See `.env.example`.
