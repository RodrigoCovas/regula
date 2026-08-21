## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context layout with one root `CONTEXT.md` and ADRs under `docs/adr/`. See `docs/agents/domain.md`.

### Checks

Run after code changes, before committing:

- Tests: `python -m pytest backend/tests/ -q`
- Type checking: `mypy` (config in `pyproject.toml`, deps in `backend/requirements-dev.txt`)
