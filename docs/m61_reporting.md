# M6.1: Offline reporting and paired uncertainty

M6.1 **does not rerun estimators**. It audits an M6.0 `m512-support` manifest,
checks each listed artifact's SHA-256, CSV headers and row count, and computes
coverage and paired squared-error comparisons directly from `trials.csv`.

```bash
python -m opebenchlab.m61_report \
  --input-dir reports/m60_support \
  --output-dir reports/m61_support \
  --bootstrap 2000 --seed 42
```

Outputs: `coverage.csv`, `paired_bootstrap.csv`, and `report.json`.

- **Coverage** is finite estimates divided by unique trial IDs per configuration.
  Undefined or absent estimates count as failures. Conditional RMSE only uses
  finite estimates; interpret it alongside coverage.
- **Paired differences** are contextual-DR squared error minus SNIPS squared
  error on identical `(requests, exploration, trial)` records. Negative favors
  DR. Nonfinite or missing pairs are excluded and counted.
- **Uncertainty** is a deterministic, paired nonparametric percentile bootstrap
  of the mean squared-error difference within each configuration. These are
  exploratory 95% intervals, **not** simultaneous confidence bands, and are
  not calibrated for adaptive model selection or multiple comparisons.
- **Integrity** is verified against the original manifest, not independently
  certified by a trusted external source. The report records its hash.

This tool currently supports M5.12-format trial records and manifests only;
other registered benchmarks have different schemas. No claim is made about
confidence-interval coverage of the underlying OPE estimators, which remains
future work. Bootstrap resampling assumes trial independence within a setting.
