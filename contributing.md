# Contributing

whichvlm is a Python 3.11+ CLI. It detects local hardware, fetches VLM
metadata, ranks compatible model artifacts, and prints CLI, JSON, or markdown
output.

## Setup

```bash
git clone https://github.com/ptxv/whichvlm.git
cd whichvlm
uv sync --group dev
uv run whichvlm --help
```

## Issues

Search existing issues before filing a duplicate.

For hardware detection bugs, include:

- OS and Python version
- GPU or SoC model
- driver/runtime version when known
- the exact `whichvlm` command
- the relevant command output or traceback

For ranking bugs, include:

- the command and flags used
- actual top result
- whether the result used real or simulated hardware

## Testing

whichvlm uses `pytest` to test the codebase.

```bash
# Install the development dependencies used by the test suite.
uv sync --group dev

# Run all tests.
uv run pytest tests/

# Run one test file with detailed output.
uv run pytest -s -v tests/test_ranker.py
```

Before opening a pull request, include the tests you ran. For most changes, run
the full suite and compile check:

```bash
uv run pytest -q
uv run python -m compileall -q src tests
```

## Code Changes

- Add or update tests for changed behavior.
- Update output tests when changing table columns, markdown text, JSON fields,
  diagnostics, or error messages.
- Add dependencies only when the standard library and current dependencies are
  not enough.

## Pull Requests

- Use a direct title, such as `Fix AMD shared-memory detection`.
- Link the issue when one exists.
- List the tests run.
- Include sample CLI output for user-visible changes.
- Note any hardware used for manual validation.
