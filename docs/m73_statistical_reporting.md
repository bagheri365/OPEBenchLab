# M7.3 statistical reporting and reproducibility appendix

This manuscript update uses the *extended* M6.5 calibration experiment and M7.2 paired analysis. It does not change any OPE estimator or claim performance on production logs.

## Reproduce

Run the existing extended M6.5 and M7.2 protocols described in `docs/m72_validation.md`, or extract the archived `OPEBenchLab_M7.2_extended_results.zip` into the repository root. Then run:

```bash
python -m opebenchlab.m73_manuscript \
  --input-dir reports/m72_extended \
  --output-dir reports/m73_manuscript
```

Outputs: `mechanism_precision.csv` (pooled integer-count precision by mechanism), `mechanism_reversal.png` (mechanism comparison), and `evidence_manifest.json` (SHA-256 hashes, table values and prespecified difficult-regime paired comparisons). The source CSVs are external report artifacts, not checked into Git.

## Units of inference and interpretation

The original M6.5 experiment had three held-out seeds; the extended study uses eight calibration seeds and twelve disjoint held-out seeds. Thirty trials share each seed in each configuration. Seed-cluster bootstrap resamples whole seed clusters, not individual trials. Bootstrap intervals are percentile intervals from 2,000 draws; 36 paired comparisons were inspected without multiplicity adjustment. An interval excluding zero is exploratory evidence, not familywise error control.

M7.2 compares two protocols: (a) frozen calibration thresholds with potentially unequal alert rates, and (b) matched-budget ranking that flags 25% of held-out observations per configuration using **observable** scores. The matched-budget exercise uses held-out score ranks, not oracle outcome labels, to allocate alerts; it is not itself a pre-deployed threshold. High-error labels and their cutoffs come from simulator truth and calibration; they are unavailable in real deployment. For comparisons, report actual flagged counts and label prevalence alongside precision.

Cross-mechanism transfer applies pairwise-calibrated rules to **the same additive trials** already used in within-mechanism evaluation. Do not sum transfer and within rows as independent trials. In transfer, frozen validation-based thresholds can flag zero trials, making precision undefined, not zero.

SNIPS RMSE from prior milestones is conditional on the estimator being finite; report its coverage separately. No interval-coverage claim is supported by M6.2. No universal estimator ranking or reliable deployment alarm follows from these synthetic experiments.

## Claims supported by the extended results

Across 4,320 unique within-mechanism held-out trials, 1,080 alerts per warning at matched 25% budget yield 307 ESS, 298 validation, and 287 combined true positives. Thus pooled precision is 28.43%, 27.59%, and 26.57%, respectively. At 500 requests and exploration 1.0, the contextual-pairwise regime favors ESS over validation and combined in unadjusted paired seed-cluster intervals. The additive regime reverses the relative usefulness of ESS and validation RMSE. This mechanism dependence is a descriptive finding of this simulator family, not evidence of general transferability.
