# Contributing to nthlayer-common

Thank you for considering contributing to **nthlayer-common** — the shared
library for the NthLayer ecosystem (LLM interface, provider infrastructure,
identity resolution, error hierarchy, tier definitions, decision records,
verdicts, manifests). We're in active v1.5 development and welcome feedback
from the SRE/DevOps community.

This library is consumed by every ecosystem member except `opensrm`, so
changes here ripple downstream — keep the public API stable and well-tested.

## Ways to Contribute

- **Report bugs / request features** — [open an issue](https://github.com/rsionnach/nthlayer-common/issues).
- **Discuss** — [GitHub Discussions](https://github.com/rsionnach/nthlayer/discussions) for the wider ecosystem.
- **Code & docs** — pull requests welcome (see below).

## Development Setup

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the venv
git clone https://github.com/rsionnach/nthlayer-common.git
cd nthlayer-common
uv sync --extra dev                  # creates .venv with test/lint tools

# Run the test suite
uv run pytest                        # full suite
uv run pytest tests/test_<name>.py   # a single file
uv run pytest -k "<expr>"            # by name

# Lint
uv run ruff check src/ tests/
```

> **Python.** Requires Python 3.11+ (`uv` will provision it via
> `uv python install` if needed). Unlike the rest of the ecosystem,
> `nthlayer-common` has no sibling dependencies — it's the shared base.

A clean clone to a green `uv run pytest` should take well under five minutes.

## Pull Request Process

1. Fork the repository and create a feature branch off `main`
   (`git checkout -b feat/your-change`).
2. Make your change with tests.
3. Ensure tests pass: `uv run pytest`.
4. Ensure lint passes: `uv run ruff check src/ tests/`.
5. Commit using Conventional Commits (see below).
6. Push to your fork and open a PR against `main`.

Commits land on `main`; `release-please` maintains the release PR and cuts
versioned releases from your conventional commits.

## Development Guidelines

### Code Style

- Python 3.11+, type hints required (the package ships `py.typed`).
- Ruff for linting. The lint floor is **frozen** at
  `select = ["E4","E7","E9","F","I","UP","SIM","B"]` (ecosystem ruff-floor
  parity). `E501` and the full `W` family are deliberately **not** in the
  floor — do not add them without ecosystem-wide alignment.
- No `per-file-ignores`: keep all imports above any `pytestmark` /
  `pytest.importorskip` block so `E402` doesn't fire.

### Commit Messages

```
<type>: <description>

<optional body>
```

`feat` / `fix` / `perf` / `deps` / `refactor` / `docs` surface in the
changelog; `chore` / `test` / `ci` / `build` / `style` are hidden.

### Testing

- Add tests for new behaviour; keep the public API covered.
- A Docker-based smoke gate (`tests/smoke/test_imports.py`) runs against the
  freshly built wheel before each PyPI publish — keep imports clean.

## Finding Something to Work On

Browse [open issues](https://github.com/rsionnach/nthlayer-common/issues) and
look for `good-first-issue` / `help-wanted` labels. Maintainers track detailed
work in **Beads**, a Dolt-backed board in the `opensrm` repo
(`cd ../opensrm && bd ready --json`) — you don't need it to contribute.

## Code of Conduct

Be respectful and constructive — we're all here to build better reliability
tooling.

## Questions?

- [GitHub Issues](https://github.com/rsionnach/nthlayer-common/issues) — bugs and features.
- [GitHub Discussions](https://github.com/rsionnach/nthlayer/discussions) — general questions.

## License

nthlayer-common is distributed under the Apache License 2.0. By contributing,
you agree that your contributions will be licensed under the same terms (see
`LICENSE`).

---

**Thank you for helping make NthLayer better!**
