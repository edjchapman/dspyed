# Contributing

Solo-maintained portfolio project — the process is part of the product.

## Flow

1. Branch from `main` (never commit to `main` directly — it's protected).
2. Make changes; `make check` must pass locally (the pre-commit hook runs it too).
3. Open a PR. CI runs `make check` + PR-title validation as required checks.
4. **Squash-merge.** The PR title becomes the permanent commit subject, so it
   must follow the commit standard below. Branch commits are disposable WIP.

## Commit style (strict)

`type(scope): imperative subject` — validated by `scripts/check-commit-msg.sh`
in strict mode (non-conforming subjects are rejected, in the hook and in CI).
Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
Conventional commits feed release-please: `feat:` / `fix:` subjects cut releases.

## Quality gate

`make check` = markdown link/anchor validation + `ruff format --check` +
`ruff check` + `pyright` + `pytest -m "not llm"` (offline — sockets disabled).
Tests that hit a live LLM carry `@pytest.mark.llm` and never run in CI.
