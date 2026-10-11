# M7.0 — Evidence ledger

M6.1–M6.2 results were subsequently checked against the recovered `reports.zip` archive in M7.1; M6.3–M6.5 were checked against their milestone archives. The experiment CSVs are **not** included in the Git source ZIP. See `m71_figures.md` for source paths, figure regeneration, and SHA-256 manifest generation.

## M6.1: paired comparisons and finite-estimate coverage (re-audited M7.1)

Source files: `reports/m61_support/coverage.csv` and `reports/m61_support/paired_bootstrap.csv`.

- SNIPS: 359/360 finite estimates; support-aware: 360/360.
- At exploration 1.0, the mean DR minus SNIPS squared-error difference is +0.006750 at 500 requests (29 finite pairs; bootstrap 95% CI [+0.002338, +0.012393]) and +0.000846 at 2,000 requests (30 finite pairs; CI [+0.000417, +0.001326]). Both favor SNIPS.
- Only 2/12 unadjusted paired intervals exclude zero; these do not establish multiplicity-adjusted significance.

## M6.2: replication and stress tests (re-audited M7.1)

Source files: `reports/m62_replication/robustness_summary.csv`, `reports/m62_support/robustness_summary.csv`, and `reports/m62_mechanism/robustness_summary.csv`.

| Suite | Setting (exploration 1.0) | SNIPS conditional RMSE | Contextual DR RMSE | SNIPS coverage |
|---|---|---:|---:|---:|
| Replication | n=500 | 0.03620 | 0.09506 | 93.3% |
| Replication | n=2000 | 0.01777 | 0.03557 | 100% |
| Replication | n=10000 | 0.00735 | 0.00656 | 100% |
| Mechanism: additive | n=500 | 0.03617 | 0.03186 | 91.7% |
| Mechanism: contextual-pairwise | n=500 | 0.03617 | 0.09749 | 91.7% |

| Support stress (n=500, exploration 1.0) | Mean test ESS | SNIPS coverage | SNIPS conditional RMSE | Support-aware RMSE |
|---|---:|---:|---:|---:|
| 5 candidates, K=2 | 8.30 | 100% | 0.02495 | 0.02495 |
| 5 candidates, K=3 | 2.85 | 95% | 0.04251 | 0.05786 |
| 5 candidates, K=4 | 1.55 | 70% | 0.04842 | 0.09584 |
| 7 candidates, K=3 | 0.80 | 40% | 0.05370 | 0.14751 |

Replication has 90 trials per configuration; support stress has 20 trials per configuration. RMSE for SNIPS conditions on finite estimates. The M6.2 interval-coverage outputs have zero computed intervals where evaluation influence standard errors are unavailable, and cannot support coverage claims.

## M6.3: reward-model generalization and DR decomposition

| Mechanism | Requests | Exploration | Test ESS | DM bias | Correction mean | DR RMSE | Target prediction RMSE | Logged prediction RMSE | SNIPS coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| additive | 500 | 1.0 | 2.5833 | 0.0032 | -0.0056 | 0.0319 | 0.0147 | 0.0152 | 91.7% |
| additive | 2000 | 1.0 | 11.8667 | 0.0019 | -0.0030 | 0.0152 | 0.0084 | 0.0083 | 100.0% |
| additive | 10000 | 1.0 | 56.3333 | -0.0003 | -0.0009 | 0.0062 | 0.0081 | 0.0083 | 100.0% |
| contextual-pairwise | 500 | 1.0 | 2.5833 | -0.1619 | 0.1371 | 0.0975 | 0.1697 | 0.0435 | 91.7% |
| contextual-pairwise | 2000 | 1.0 | 11.8667 | -0.0968 | 0.0978 | 0.0353 | 0.1106 | 0.0333 | 100.0% |
| contextual-pairwise | 10000 | 1.0 | 56.3333 | -0.0201 | 0.0185 | 0.0060 | 0.0264 | 0.0160 | 100.0% |

DR identity (trial-level errors): `e_DR = e_DM + C`, hence `E[e_DR²] = E[e_DM²] + E[C²] + 2E[e_DM C]`. In the M6.3 summary, `cross_second_moment` stores the **full** `2E[e_DM C]` contribution, not `E[e_DM C]`. The covariance is separately reported. Zero target matches imply zero empirical correction under the specified full-slate weighting, not universal failure of all DR estimators.

## M6.4: retrospective warning ranking

| Warning | Mean precision across 12 cells |
|---|---:|
| ridge_target_range | 18.3% |
| split_target_disagreement | 21.1% |
| test_ess | 26.8% |
| validation_gap | 26.7% |
| validation_rmse | 28.3% |

Each cell labels the 15/60 trials with the largest absolute DR errors; 25% is the within-cell prevalence. The retrospective warning thresholds are not prospective calibration.

## M6.5: frozen calibration and held-out evaluation

| Evaluation | Warning | Trials | Alerts | High errors | True positives | Precision | Recall | False-positive rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| within | ess | 1080 | 320 | 272 | 83 | 25.9% | 30.5% | 29.3% |
| within | validation | 1080 | 280 | 272 | 71 | 25.4% | 26.1% | 25.9% |
| within | combined | 1080 | 293 | 272 | 70 | 23.9% | 25.7% | 27.6% |
| pairwise-to-additive | ess | 540 | 160 | 78 | 20 | 12.5% | 25.6% | 30.3% |
| pairwise-to-additive | validation | 540 | 0 | 78 | 0 | undefined | 0.0% | 0.0% |
| pairwise-to-additive | combined | 540 | 1 | 78 | 0 | 0.0% | 0.0% | 0.2% |

Pooled rates are computed from integer counts, not averages of configuration-level rates. Transfer evaluations reuse additive held-out trials and must not be counted as additional independent observations. Warning thresholds and oracle outcome cutoffs were frozen using calibration seeds; the oracle labels do not enter warning decisions. Actual held-out alert rates differ between warning rules, so raw recall is not a matched-budget ranking comparison.

## Evidence boundaries

- The original M6.1 and M6.2 CSV outputs have now been independently re-audited from `reports.zip`; see `m71_figures.md` for regeneration and source-file hashing.
- Simulator truth is available only for retrospective validation, not deployment-time warning computation.
- Three held-out seed clusters per configuration are insufficient for precise seed-level uncertainty quantification.
- These studies do not establish performance on production randomized recommendation logs.
