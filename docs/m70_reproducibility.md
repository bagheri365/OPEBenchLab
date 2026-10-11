# M7.0 reproducibility and source audit

## Environment

From a clean checkout, use Python 3.11 (or another version supported by `pyproject.toml`):

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

## Experiment entry points

```bash
python -m opebenchlab.m60_benchmark list
python -m opebenchlab.m61_report --help
python -m opebenchlab.m62_robustness --help
python -m opebenchlab.m63_diagnostics --help
python -m opebenchlab.m64_risk --help
python -m opebenchlab.m65_calibration --help
```

The exact benchmark invocations and output schemas are documented in the corresponding `docs/m6*.md` files. For example, the M6.5 held-out experiment can be reproduced with:

```bash
python -m opebenchlab.m65_calibration \
  --trials 30 --requests 500 2000 10000 \
  --explorations 0.5 1.0 \
  --calibration-seeds 42 1729 31415 \
  --heldout-seeds 2718 1618 12345 \
  --alert-fraction 0.25 --bootstrap 500 \
  --output-dir reports/m65_calibration
```

## Evidence provenance

M7.0's `m70_evidence.md` was cross-checked against three separately uploaded result archives:

- `m63_diagnostics.zip`: `summary.csv`, `components.csv`, `protocol.json`.
- `m64_risk.zip`: `risk_summary.csv`, `diagnostics.csv`, `protocol.json`.
- `m65_calibration.zip`: `calibration.json`, `heldout_predictions.csv`, `heldout_summary.csv`.

The Git source ZIP contains implementation and protocol documentation but **does not contain these run outputs**. For an externally reproducible release, archive the raw results, code commit hash, command lines, dependency versions, seeds, and checksums in a versioned release artifact. Earlier M6.1/M6.2 results were not available for direct re-audit in this synthesis.

## Numerical and statistical audit checklist

1. Count independent held-out **trials** once per mechanism and configuration; transfer rows reuse the same additive trials.
2. Compute pooled precision as total true positives divided by total alerts; do not average per-cell precision. Define precision as unavailable when there are zero alerts.
3. Record high-error prevalence and realized alert rate for every transfer or within-mechanism comparison.
4. Keep target-reward oracle metrics and oracle outcome labels outside observable warning inputs.
5. Report both SNIPS coverage and conditional accuracy; do not treat undefined estimates as zero-error trials.
6. For M6.3, verify the full cross contribution `2 E[e_DM C]` when reconstructing DR MSE.
7. Interpret seed-cluster bootstrap intervals cautiously with only three held-out seeds.
8. Do not claim matched held-out alert budgets solely from a fixed calibration quantile.

## Release checklist

Before submitting a paper, rerun or retrieve M6.1 and M6.2 output archives, increase independent held-out seeds, freeze a tagged source revision, and generate figures directly from versioned output tables. This milestone intentionally does not manufacture missing experiment files or assert results for unaudited outputs.
