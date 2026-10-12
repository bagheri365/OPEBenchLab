# v1.0.0 release checklist

## Verify the source

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m opebenchlab.m60_benchmark list
python -m opebenchlab.m72_validation --help
python -m opebenchlab.m73_manuscript --help
```

Check that the test suite passes and the listed CLI entry points respond. These checks establish packaging and smoke-test health, **not** statistical validity.

## Reproduce the reported outputs

- Recover the original M6.1–M6.5 and M7.2 report archives from the research archive; they are not bundled with this Git source release.
- Follow `docs/m70_reproducibility.md`, `docs/m71_figures.md`, `docs/m72_validation.md`, and `docs/m73_statistical_reporting.md` for experiment and figure commands.
- Verify each available `figure_manifest.json` or `evidence_manifest.json` input checksum against its corresponding archived CSV. Record archive download location and SHA-256 in the GitHub release notes; **do not claim the dataset is published until an accessible archive exists**.
- Read `docs/m81_release_audit.md` before citing confidence intervals or pooled results.

## Release commands (run after review)

```bash
git status --short
python -m pytest -q
git add README.md CHANGELOG.md pyproject.toml docs/m70_research_draft.md docs/m73_statistical_reporting.md docs/m81_release_audit.md docs/m81_release_checklist.md
git commit -m "chore: finalize v1.0.0 research release"
git push origin main
git tag -a v1.0.0 -m "OPEBenchLab v1.0.0 research release"
git push origin v1.0.0
```

Create a GitHub release from `v1.0.0` with the changelog, caveats, source revision, and links/checksums for **actually published** research archives. Confirm the tag does not already exist before creating it. Tag only after local validation.
