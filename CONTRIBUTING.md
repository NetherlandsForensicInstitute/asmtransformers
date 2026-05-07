# Contributing

ASMTransformers is organized as a monorepo:

- `asmtransformers/`: Python training, preprocessing, inference, and model code.
- `citatio/`: Python FastAPI service that depends on `asmtransformers`.
- `sententia/`: Java/Ghidra extension that talks to the Citatio API.
- `examples/`: example projects and demo data.

Most of this guide applies to all contributors. The final section adds repository-specific guidance for coding agents.

## Development Principles

- Read the relevant code before making changes. Follow the style already present in each file.
- Keep changes small, direct, and narrowly scoped to the requested behavior.
- Prefer small, direct functions over broad abstractions.
- Preserve current API behavior unless the change intentionally updates it.
- Avoid introducing heavy dependencies unless they are necessary for the requested change.
- Add comments only where the logic is non-obvious.
- Avoid reformatting or rewriting unrelated code.

## Python Development

The Python projects use PDM and are managed independently from their own directories.

For `asmtransformers`:

```bash
cd asmtransformers
pdm install
```

For `citatio`:

```bash
cd citatio
pdm install
```

Both Python projects currently require Python `>=3.13,<3.14`.

Use the project-local PDM scripts:

```bash
pdm run check
pdm run test
pdm run all
```

`pdm run check` validates the lockfile and runs Ruff format/lint checks. `pdm run test` runs the test suite
with coverage collection. `pdm run all` is the pre-PR convenience command; it currently runs `check`, `test`, and
the coverage report.

Use `pdm run fix` only when you intentionally want Ruff to rewrite formatting or lint issues.

Python style is configured in each package's `pyproject.toml`:

- Ruff format and lint are used.
- Single quotes are preferred.
- Line length is 120 characters.
- Tests live in each package's `tests/` directory.

When changing shared behavior between `asmtransformers` and `citatio`, run checks and tests in both directories.

## Java/Ghidra Development

The Ghidra extension lives in `sententia/` and is built with Gradle from that directory:

```bash
cd sententia
gradle -PGHIDRA_INSTALL_DIR=/path/to/ghidra
```

`GHIDRA_INSTALL_DIR` must point to a local Ghidra installation. The Gradle file includes a Ghidra template section marked as not to be modified; avoid changing that section unless the Ghidra build integration itself is the target of the work.

## API And Behavior Changes

Be explicit when a change affects any public or cross-component contract, including:

- Python package imports or public classes/functions.
- FastAPI request or response shapes.
- SQLite schema or stored data format.
- Model/tokenizer behavior or serialized model assets.
- Sententia-to-Citatio API interactions.

If a behavior change is intentional, update or add tests and documentation that make the new behavior clear.

## Testing Expectations

For Python changes, run the commands that match the scope of your change:

- `pdm run test` while validating behavior changes.
- `pdm run check` while validating style, lockfile, and lint changes.
- `pdm run all` in each affected Python project before opening a pull request.

For documentation-only changes, runtime tests are not normally required.

For Sententia changes, build the extension when a Ghidra installation is available. If it is not available, state that the Gradle build was not run.

## Pull Request Checklist

Before opening a pull request, check that:

- The change is focused and described clearly.
- Relevant tests and checks have been run, or any skipped checks are explained.
- User-facing behavior changes are documented.
- Public API, schema, model, or dependency changes are called out explicitly.

## Notes For Coding Agents

When using this file as a stand-in for `AGENTS.md`, follow the contributor guidance above and keep these additional rules in mind:

- Determine which subsystem owns the behavior before editing, and avoid cross-repo churn.
- Respect the current worktree. The repository may contain user-created files, generated outputs, virtual environments, coverage files, IDE metadata, and caches. Leave unrelated files alone.
- Ask before choosing a direction that would change behavior, add dependencies, modify data formats, or remove compatibility.
