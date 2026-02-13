# Contributing

Thanks for contributing to the Banana Disease Detection project. This doc covers how to run tests, branch naming, PR checklist, and code style. **CI/CD is planned** (see [docs/enhancement/implementation/ENHANCEMENT_3_DEVOPS_CICD.md](docs/enhancement/implementation/ENHANCEMENT_3_DEVOPS_CICD.md)).

---

## Running tests

```bash
# Install dependencies (use uv if available)
uv sync

# Run all unit tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=0

# Run integration tests (if any; may require DB or env)
uv run pytest tests/integration/ -v
```

Coverage is optional; use `--cov-fail-under=0` to avoid failing until you set a target. See [README](README.md) for more test commands.

---

## Branch naming

- `feature/short-description` — new feature
- `fix/short-description` — bugfix
- `docs/short-description` — documentation only
- `chore/short-description` — tooling, deps, no user-facing change

---

## PR checklist

- [ ] Unit tests pass: `uv run pytest tests/unit/ -v`
- [ ] Code style: run formatter/linter (see below)
- [ ] No secrets in code; use `.env` and never commit `.env`
- [ ] API auth is not in scope for this repo; document in deployment if you add it (see [docs/TODO_LIST.md](docs/TODO_LIST.md))

---

## Code style

- **Formatter:** Use [ruff](https://docs.astral.com/ruff/) or [black](https://black.readthedocs.io/) for Python (e.g. `ruff format .` or `black .`).
- **Linter:** `ruff check .` is recommended.
- **Line length:** 100 or 88 (match black default if using black).
- **Docstrings:** Use for public functions and modules; plain English is fine.

Example (optional) in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
```

---

## Environment and secrets

- Copy `.env.example` to `.env` and fill in values locally. **Never commit `.env`.**
- In production, use environment variables or a secret manager; see [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

---

## CI/CD

Automated CI/CD (build, test, deploy) is **planned**; see [ENHANCEMENT_3_DEVOPS_CICD.md](docs/enhancement/implementation/ENHANCEMENT_3_DEVOPS_CICD.md). Retrain scheduling (cron, Task Scheduler) is documented in the project docs and `scripts/run_retrain.py`.
