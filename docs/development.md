# Development

How this repository is kept reproducible. The rules are short; the checks enforce most of them.

## Setup

```bash
uv sync                      # core environment (locked)
uv run pre-commit install    # lint, large-file and format checks on every commit
```

## Rules

1. **New work goes in `experiments/<study>/`** with a README naming the question, scripts, outputs and
   paper item. No scripts at the top level (a test enforces this).
2. **Paths come from `spindlebench.paths`.** Nothing may depend on the working directory (a test
   enforces this for string literals).
3. **Every experiment script calls `provenance.track(__file__, protocols=[...])`** after its imports.
   It writes `results/provenance/<study>/<script>.json`: commit, uncommitted-changes flag, package versions,
   arguments, protocol versions, run time.
4. **Scoring rules are versioned.** A change to a matching rule, tolerance, scope or data cut that can
   move F1 gets a new `VERSION` in `src/spindlebench/protocols/` and a line in `CHANGELOG.md`.
5. **Detectors follow `detectors/INTERFACE.md`** and carry a `manifest.json`; `tests/test_detectors.py`
   checks every stored detection file.
6. **Refactors change no number.** Before a structural change, freeze a snapshot of the results; after
   it, re-run and compare with `scripts/compare_snapshot.py`. Intended changes are listed with a reason
   in the snapshot's `accepted.tsv` and in `CHANGELOG.md`.
7. **Licences.** Code MIT; docs and results CC BY 4.0; GPL ports keep their SPDX header. `REUSE.toml`
   covers every file (`uvx --from "reuse[charset-normalizer]" reuse lint`). Nothing derived from DREAMS
   beyond scores is published (DATA_SOURCES.md).
8. **Commits are small and explained.** One logical step per commit; results regenerated in the same
   commit as the code that changed them.

## Checks

```bash
uv run ruff check .
uv run pytest                                   # tests marked `data` skip without the local cache
uv run python scripts/compare_snapshot.py       # after a refactor
uvx --from "reuse[charset-normalizer]" reuse lint
uvx cffconvert --validate -i CITATION.cff
```

CI (`.github/workflows/ci.yml`) runs lint and tests on Linux and Windows, and the REUSE and citation
checks.
