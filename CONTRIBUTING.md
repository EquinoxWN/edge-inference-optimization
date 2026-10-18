# Contributing

Thanks for helping. This is a Python project (PyTorch, transformers); issues and pull requests are
welcome.

## Set up and check your change

Needs Python 3.11+ and make; the GPU run needs an NVIDIA GPU. Use a virtual environment;
downloads, caches and models stay in git-ignored `.tmp/`.

```bash
make setup       # CPU torch and dev tools
make lint        # ruff, ruff format --check, mypy --strict
make test        # tests on a tiny random model, no download
make audit       # pip-audit
make setup-gpu && make run   # the GPU comparison; deletes the model afterwards
```

A pull request is ready when `make lint` and `make test` pass and CI is green.

## Guidelines

- Changes to the measurement (`bench.py`) need a fresh `make run`, with the new JSON in
  `docs/results/runs/` and `docs/results/m1.md` updated; say in the pull request what changed.
- New downloads must be pinned in `sources.py` by revision, size and SHA-256.
- Never commit model files; `make purge` removes anything left in `.tmp`.
- Add or update a test for every behaviour change.
- Significant design changes need an ADR in `docs/adr/` (copy `0001` for the format).
- Keep function comments to one short phrase; let names and tests explain the rest.
- Report security problems privately, as described in [SECURITY.md](SECURITY.md).

By contributing you agree that your work is released under the [MIT License](LICENSE).
