# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations, run inside the clone so `gh` infers the repo automatically.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments with `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels --jq '[.[] | {number, title, labels: [.labels[].name], body}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."` — the label vocabulary lives in `docs/agents/triage-labels.md`.
- **Close**: `gh issue close <number> --comment "..."`

## Issue shapes used in this repo

- **Specs**: a parent issue carrying the problem statement, user stories, implementation and testing decisions, and out-of-scope notes. Child tickets link back with a `## Parent` section at the top of the body.
- **Tickets**: a `## What to build` section and `## Acceptance criteria` as checkboxes, plus a `## Blocked by` section naming the blocking issues (`None (can start immediately)` when unblocked). A ticket is workable once every blocker is closed and it carries `ready-for-agent`.
- **Work ties back to tickets**: the commit message names the ticket it closes (`(#<n>)`), and the issue closes once its acceptance criteria are verified.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either — resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## Pull requests

**PRs as a request surface: no.** Work arrives as issues. Should that ever change, external PRs would run through the same triage labels as issues, using the `gh pr` equivalents: `gh pr view <number> --comments` and `gh pr diff <number>`, `gh pr list --state open`, then `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.
