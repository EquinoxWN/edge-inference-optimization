# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub:
**Security > Report a vulnerability** on this repository
(`https://github.com/EquinoxWN/edge-inference-optimization/security/advisories/new`). Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect. You will get an answer
within 7 days. Fixes are released as soon as they are ready, and you are credited unless you ask
not to be.

## Supported versions

This project is pre-1.0. Only the latest commit on `main` receives fixes.

## Scope

- **In scope:** downloads that could be swapped or tampered with, loading untrusted model or data
  files, writing outside the repository, and leaking anything from the environment.
- **Out of scope:** model accuracy, and vulnerabilities in PyTorch, transformers or bitsandbytes
  themselves (report those upstream; the CI audit tracks them).

## How this repository protects itself

- Every download is pinned to a model or dataset revision, an exact size and a SHA-256, fetched
  over HTTPS only, and deleted if the check fails.
- Weights are loaded from safetensors (no pickle); the held-out parquet is read with an explicit
  column list and every row is validated.
- Hugging Face runs offline with telemetry off; every cache and temp path points into `.tmp`;
  output paths are refused if they leave the repository.
- The model and data are deleted after each run, also when the run fails.
- Every GitHub Action is pinned to a full commit SHA, and workflows run with read-only
  permissions and without persisted credentials.
- Dependabot update pull requests are paused (`open-pull-requests-limit: 0`) so the repository keeps a single `main` branch; dependencies are updated by hand and checked by the CI audit.
- CI runs ruff, mypy --strict, the tests and pip-audit on every push and pull request.
