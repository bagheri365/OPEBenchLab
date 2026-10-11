# M6.2 robustness protocol (staged)

M6.2 reuses the M5.11/M5.12 estimators without changing their training or selection logic. The experiment grid is specified before running; do not select or modify settings based on observed RMSE.

## Suites

- **replication**: independent random seeds, same default slate environment.
- **support**: candidate/slate dimensions `(5,2)`, `(5,3)`, `(5,4)`, `(7,3)`, with exact logging propensities. This deliberately varies exact-match probability.
- **mechanism**: original contextual pairwise reward (`pair_bonus=0.2`) and additive negative control (`pair_bonus=0`). This **does not** yet test a truly misspecified interaction mechanism; it is an initial mechanism sensitivity check.

The source implementation exhaustively enumerates context/slate combinations and limits `candidates <= 7`. Candidate count 10 is therefore deferred; pretending to run it with the existing implementation would be incorrect.

## Run

Quick smoke test:

```bash
python -m opebenchlab.m62_robustness --suite replication --trials 1 --seeds 42 --requests 40 --explorations 1 --ridges 1 --output-dir reports/m62_smoke
```

Staged main run (start with replication before the more expensive support suite):

```bash
python -m opebenchlab.m62_robustness --suite replication --trials 30 --seeds 42 1729 2026 --requests 500 2000 10000 --explorations 0.2 0.5 1.0 --output-dir reports/m62_replication
python -m opebenchlab.m62_robustness --suite support --trials 10 --seeds 42 1729 --requests 500 2000 --explorations 0.5 1.0 --output-dir reports/m62_support
python -m opebenchlab.m62_robustness --suite mechanism --trials 30 --seeds 42 1729 --requests 500 2000 --explorations 0.2 1.0 --output-dir reports/m62_mechanism
```

## Outputs and limitations

- `robustness_trials.csv`: every estimator result, including non-finite values, keyed by suite, mechanism, dimensions, seed, request count, exploration and trial.
- `robustness_summary.csv`: estimator coverage, conditional RMSE, bias, ESS, and support-failure counts.
- `interval_coverage.csv`: explicit **unavailable** entries. M5.11 does not export evaluation-sample influence standard errors, so individual OPE confidence intervals **cannot be calibrated from the current rows**. Do not confuse bootstrap intervals for across-trial MSE differences with individual policy-value confidence intervals.
- `protocol.json`: exact parameter choices, protocol digest, analysis definitions and limitations.

The existing M5.11 seed formula depends on request count, exploration, and trial. Different top-level seeds generate different random streams, but the experiment does not assert formal independence beyond the pseudorandom generator. No multiple-comparison correction is applied. Report coverage alongside conditional RMSE and never silently drop undefined estimates.
