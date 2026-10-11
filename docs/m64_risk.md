# M6.4 — Target-distribution risk diagnostics

Run:

```bash
python -m opebenchlab.m64_risk --trials 30 --seeds 42 1729 --requests 500 2000 10000 --explorations 0.5 1.0 --ridges 0.01 1 10 --output-dir reports/m64_risk
```

This preserves M6.3's synthetic simulation, model fitting, and common random numbers.

* `diagnostics.csv`: observable signals (training/validation RMSE, generalization gap, ridge target-mean range, disagreement between two models fitted on separate training halves, validation target matches, evaluation support/ESS) and **separate oracle-only** outcomes (true value, DR error, target prediction error). No evaluation rewards or oracle values enter a warning signal.
* `risk_summary.csv`: within each mechanism/request/exploration configuration, retrospective top-quartile warnings compared with the top quartile of absolute DR error. Includes counts, precision, recall and false-positive rate. ESS uses the bottom quartile as the warning.
* `protocol.json`: configuration and explicit signal/outcome separation.

**Interpretation:** Quartile cutoffs are calculated using all trials within each configuration. This is a retrospective diagnostic audit, not an operational detector with thresholds learned on independent calibration data. Ties may cause more than 25% of trials to be flagged. Prediction disagreement is measured on evaluation contexts/actions but does not use evaluation rewards. The held-out logged validation RMSE is not target-policy prediction error. Synthetic oracle outcomes are used only to evaluate warning associations. No causal or deployment-level guarantees follow.
