# M6.5: seed-held-out warning calibration

Run:

```bash
python -m opebenchlab.m65_calibration --trials 30 --output-dir reports/m65_calibration
```

The CLI runs the M6.4 data-generating protocol independently on disjoint calibration
and held-out seed sets. The default uses three seeds each, 30 trials per seed,
three request counts, two exploration settings, and two reward mechanisms.

For each source mechanism/request-count/exploration configuration, calibration
trials define empirical midrank percentile scores for *low ESS* (sign reversed),
*high logged validation RMSE*, and their equal-weight average. The top 25%
score quantile is a frozen warning cutoff; strict greater-than comparison
handles ties conservatively. Calibration DR absolute-error 75th percentiles
also define the frozen **oracle outcome label** for held-out evaluation.
Neither calibration warning scores nor held-out warning decisions use oracle
outcomes or evaluation rewards. Oracle errors are used to learn the outcome
cutoff and assess performance, **not to score or flag a trial**.

Outputs:

- `calibration.json`: seeds, experiment settings, score definitions, frozen
  warning and oracle-outcome cutoffs.
- `heldout_predictions.csv`: one decision per held-out trial and rule, with
  observable scores and oracle outcome labels separated by column.
- `heldout_summary.csv`: held-out precision, recall, false-positive rate,
  actual alert rate and percentile intervals from seed-cluster bootstrap.

`within` evaluates each mechanism on new seeds; `pairwise-to-additive` applies
contextual-pairwise calibration to additive data at the same sample size and
exploration. The transfer outcome label uses the **source** mechanism's
frozen oracle error cutoff, so prevalence may shift substantially. Compare
alert rate and base prevalence alongside precision and recall.

## Limitations

- This is **simulation-supervised calibration**: oracle outcomes are available
  for fitting the calibration error cutoff. The warning itself uses only
  observable quantities. Deploying absolute-error thresholds would require
  simulation fidelity or another trustworthy reference.
- A fixed calibration score quantile does not guarantee an exact held-out alert
  budget, especially with tied or shifted ESS distributions. Report actual
  alert rates; do not compare raw recall without them.
- Percentile intervals resample entire held-out seed clusters; three default
  seeds are too few for robust confidence intervals. Use more seeds for
  publishable inference. The bootstrap is descriptive, not a formal coverage
  guarantee.
- Combining two signals by averaging percentiles is a **predeclared rule**, not
  an optimized classifier; this prevents held-out selection but may be suboptimal.
- M6.4 generates paired reward mechanisms with common random numbers. Seed
  independence is at the seed-cluster level, not at the trial level.
