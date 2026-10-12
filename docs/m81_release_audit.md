# M8.1 release audit and statistical scope

**Release disposition:** OPEBenchLab v1.0.0 is a completed **synthetic research software artifact**, not a peer-reviewed paper or a deployment-validated OPE system. This release closes documentation and reproducibility work; it does **not** repair or replace the statistical inference procedures identified below. The source estimators and simulation protocols are unchanged.

## Open statistical issues (explicitly accepted for the software release)

1. **Conditional matched-budget bootstrap.** M7.2 ranks and selects alerts once on the observed held-out sample. Its seed-cluster bootstrap resamples those *fixed decisions*. The reported intervals therefore describe variation **conditional on the observed alert selection**, not uncertainty of the full re-ranking and alert-selection procedure. A re-ranked cluster bootstrap would answer a different question; it has **not** been implemented. Do not describe existing intervals as prospective policy performance intervals.
2. **Shared seeds across configurations.** Additive and contextual-pairwise mechanisms intentionally share generated observations, and configurations can share seed identifiers. Pooled values are **descriptive**. The repository does not provide a joint-seed bootstrap across configurations; do not attach per-configuration intervals to pooled results or treat cross-mechanism rows as independent replications.

## Other limitations

- The 36 M7.2 paired intervals are unadjusted for multiplicity and should be interpreted as exploratory. Twelve independent held-out seed clusters improve replication but do not establish general external validity.
- Matched-budget alerts are selected transductively using held-out **observable** scores; they are not a frozen deployment rule. High-error outcomes use simulator truth and are not observable in production.
- SNIPS RMSE is conditional on finite estimates and must be accompanied by coverage. No interval-coverage or real-world policy-value guarantee is claimed.
- Reward mechanisms and logging policies come from one controlled synthetic simulator family. Original CSV archives are external to Git; checksummed manifests do not make the underlying data independently downloadable.

## Interpretation and closure

The verified numerical findings in `docs/m70_evidence.md` remain **descriptive evidence**. The strongest regime-specific matched-budget comparison favors ESS in the low-support contextual-pairwise setting, while the additive mechanism can favor validation RMSE. The combined score has no consistent pooled advantage. These observations do not justify a universal warning or estimator ranking.

**For a later peer-reviewed submission**, separately implement and test a re-ranked seed-cluster bootstrap, a pooled joint-seed uncertainty analysis, a multiplicity strategy, and publicly retrievable versioned data artifacts. These are **deferred research tasks**, not completed release features.

Source of findings: the M8.0 focused independent audit of committed M7.3 source. No claim of complete formal verification is made.
