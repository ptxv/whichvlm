# Contributing

Work from an issue. Fork the repo, clone your fork, make the change, and open a
pull request back to this repo.

Issue titles should start with `[bug]`, `[feat]`, or `[question]`. Bug reports
must include steps to reproduce, the command you ran, the expected result, and
the actual result.

Pull request titles should start with `[feat]`, `[bugfix]`, or the module being
changed. Keep each PR scoped to one issue. Do not mix features, fixes, and
cleanup.

Run the relevant tests before opening a PR. Use focused tests for small changes.
Use broader tests when changing ranking, fetching, runtime, output, or hardware
detection. List the exact test commands in the PR description.

Keep PR descriptions short. Say what changed, why it changed, and what remains
untested. For hardware, runtime, or performance claims, include the device, OS,
command, and observed output.

Write plainly. Keep sentences short. Prefer concrete details over filler. This
makes human review easier.

## Release

Before the first release, configure a PyPI trusted publisher for
`ptxv/whichvlm`, workflow `release.yml`, and environment `pypi`.

To publish a release, update the version in `pyproject.toml`, merge the change,
then push a matching `vX.Y.Z` tag. The release workflow builds and tests the
wheel and source distribution, publishes them to PyPI, and creates a GitHub
release with generated notes and both artifacts.
