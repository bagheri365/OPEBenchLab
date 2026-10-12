# M7.2: paired warning validation

M7.2 adds an **offline analysis** of M6.5 held-out trial predictions; it does not rerun the simulator or change estimators. To increase independent replication, first rerun `python -m opebenchlab.m65_calibration` with **more disjoint held-out and calibration seeds**, then analyze its output:

```bash
python -m opebenchlab.m72_validation --input-dir reports/m65_calibration --output-dir reports/m72_validation --bootstrap 1000
```

Outputs: `warning_summary.csv` (frozen-threshold and matched-budget metrics), `paired_precision.csv` (combined−ESS and validation−ESS differences, seed-cluster bootstrap intervals), `warning_decisions.csv` (auditable trial-level decisions), `protocol.json`.

**Estimands:** Frozen thresholds test deployment-style generalization and allow alert rates to drift. Matched-budget comparisons instead select the same number of alerts for each signal **within each held-out configuration**, based on the already frozen risk scores. The budget uses observable held-out score ranks, but never DR errors; therefore this is a transductive ranking evaluation, **not** a prospective deployment policy. Cross-mechanism transfer remains a distinct stratum.

**Uncertainty:** Resample independent held-out seed clusters, preserving paired signal decisions within each cluster. Intervals condition on the original selected alerts and do not re-rank after bootstrap resampling. With only three held-out seeds, intervals remain unstable. Per-configuration intervals do not adjust for multiple testing. The fixed number of alerts does not ensure equal alert rates in bootstrap resamples, although the full held-out dataset has matched rates.

**Limitations:** More trials under the same seed do not substitute for more independent seeds. This analysis cannot itself create new independent replication; run M6.5 with additional seeds to strengthen inference. The M6.5 high-error oracle cutoff remains calibration-only.
