# M7.1 — Audited evidence and publication figures

The archived `reports.zip` was inspected independently of the M7.0 narrative. The original M6.1 and M6.2 outputs were recovered, resolving the two outstanding evidence-source gaps. This milestone adds a deterministic plotting CLI, not new OPE estimators.

## Reproduce the figures

Extract `reports.zip` so that `reports/m61_support/`, `reports/m62_replication/`, `reports/m62_support/`, `reports/m62_mechanism/`, and `reports/m65_calibration/` exist. From the repository root:

```bash
python -m pip install matplotlib
python -m opebenchlab.m71_figures --reports-dir reports --output-dir reports/m71_figures
```

The four PNGs are `m71_replication_rmse.png`, `m71_support_coverage.png`, `m71_mechanism_rmse.png`, and `m71_heldout_precision.png`. The generated `figure_manifest.json` contains input SHA-256 digests, row counts, derived values, and interpretive cautions. **Do not commit the private or unversioned report archive as though it were part of the source tree.** For a publication release, archive the precise input files and hashes alongside the figures.

## Interpretation

- M6.1: 359/360 SNIPS finite versus 360/360 support-aware finite. Two of twelve unadjusted paired-bootstrap 95% intervals exclude zero, both favoring SNIPS at exploration 1.0 and requests 500 or 2,000. The 500-request comparison has 29 paired finite trials.
- M6.2 replication: SNIPS conditional RMSE 0.03620, 0.01777, 0.00735 at 500, 2,000, 10,000 requests (exploration 1.0); contextual DR RMSE 0.09506, 0.03557, 0.00656 respectively. The first SNIPS value conditions on defined estimates.
- M6.2 support stress (500 requests, exploration 1.0): SNIPS finite coverage declines from 100% (5 candidates, K=2) to 40% (7 candidates, K=3). The support-aware estimator remains finite in all trials but does not maintain the same accuracy.
- M6.2 additive negative control: at 500 requests and exploration 1.0, contextual DR RMSE is 0.03186 (additive) versus 0.09749 (contextual-pairwise). This is not a general misspecification experiment.
- M6.5: the plotted precision values are pooled from integer counts on held-out trials, not averages of per-cell precision. Actual alert rates are **not** equal across warnings; the chart is descriptive, not a matched-alert-budget comparison.

## Statistical and provenance cautions

The M6.2 `interval_coverage.csv` files report zero computed intervals for methods without an exported evaluation influence standard error. **Do not interpret those files as empirical 95% coverage evidence.** M6.1's twelve paired intervals are unadjusted for multiple comparisons. SNIPS conditional RMSE omits undefined outcomes, so it is not directly comparable to unconditional finite-estimator risk without the coverage rate. M6.5 transfer rows reuse additive held-out trials and must not be counted as independent observations. Three independent held-out seeds do not support precise seed-cluster uncertainty estimates.
