# M6.3: contextual DR failure decomposition

This is a **diagnostic**, not a new estimator or estimator-selection rule. It
reuses the M5.11 training/evaluation split, ridge tuning, simulator, and contextual
reward model. Additive (zero interaction bonus) and contextual-pairwise rewards
share the same generated contexts, logged actions, noise, and train/test splits.

Run:

```bash
python -m opebenchlab.m63_diagnostics --trials 30 \
  --seeds 42 1729 --requests 500 2000 10000 \
  --explorations 0.5 1.0 --output-dir reports/m63_diagnostics
```

The outputs are `components.csv` (one row per mechanism/seed/trial/configuration),
`summary.csv` (grouped second moments, covariance, support and prediction errors),
and `protocol.json` (definitions and fixed settings).

For each trial, `dm_error = mean(q_target) - true_value`,
`correction = mean(w * (observed - q_logged))`, and
`dr_error = dm_error + correction`. Consequently:

`mean(dr_error^2) = mean(dm_error^2) + mean(correction^2) + 2*mean(dm_error*correction)`.

**The cross second moment is not the covariance**: it includes the product of
means. The separate `dm_correction_covariance` is the sample covariance across
trials. A large term does not demonstrate causality.

`target_prediction_rmse` and `logged_prediction_rmse` use the known noiseless
synthetic reward on held-out contexts and actions. These diagnostics would not
be observable with ordinary real-world logs. The target prediction metric is
averaged across observed evaluation contexts, while the true policy value is
an exact average over all enumerated contexts. Therefore the DM error also
contains context-sampling variation; do not equate it with pure model bias.

SNIPS coverage is reported separately; no-match trials have undefined SNIPS
but finite DR. The evaluation rewards are used for diagnostic reporting only,
not for ridge tuning or estimator selection. Pairing is within matched seeds
and trials; this report does not calculate inferential intervals.
