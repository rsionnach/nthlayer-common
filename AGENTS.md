# nthlayer-common — agent-facing commands and conventions

Shared utilities for the NthLayer ecosystem. Pure library; no console
scripts.

## Stack

- Python ≥3.11, managed via `uv`.
- Tests: `pytest`, `pytest-asyncio`.
- Lint: `ruff` (floor frozen — see below).
- Typecheck: **not configured** (no `mypy.ini`, no `pyrightconfig.json`,
  no `[tool.mypy]`/`[tool.pyright]` in `pyproject.toml`). The package
  ships `py.typed` (PEP 561) — consumers run their own typecheck against
  this package's annotations. TODO: wire mypy or pyright for in-repo CI.

## Build / test / lint commands

```bash
uv sync                                # set up .venv
uv run pytest                          # run all tests (~758 at last count)
uv run pytest tests/test_<name>.py     # single file
uv run pytest -k "<expr>"              # single test by name
uv run ruff check src/ tests/          # lint
```

Ecosystem testing conventions: [../nthlayer/docs/testing.md](../nthlayer/docs/testing.md).

## Lint floor (frozen)

```
ruff select = ["E4", "E7", "E9", "F", "I", "UP", "SIM", "B"]
```

- Frozen post-opensrm-c5j6 (ecosystem-wide ruff-floor-parity series;
  matches nthlayer-workers' opensrm-po23 floor).
- `E501` (line-too-long) and the full `W` family are **separate hygiene
  calls**, not part of the floor — do not add them to `select` without
  ecosystem-wide alignment.
- **No `per-file-ignores`.** Tests place all imports above
  `pytestmark` / `pytest.importorskip` blocks, so `E402` doesn't fire.
  Keep that layout when adding new tests.

## Conventional Commits + release-please

Commits land on `main`; `release.yml` + `release-please-action@v4`
maintain the release PR. Taxonomy:

- Surfaces in changelog: `feat` / `fix` / `perf` / `deps` / `refactor` / `docs`
- Hidden: `chore` / `test` / `ci` / `build` / `style`

PyPI publish via trusted-publishing; a Docker-based smoke gate runs
`tests/smoke/test_imports.py` against the freshly-built wheel before
publish.
