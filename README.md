# OPEBenchLab

**When can off-policy estimates of recommendation slates be trusted?**

OPEBenchLab is a controlled synthetic research project for studying off-policy evaluation (OPE) of contextual recommendations and ordered top-K slates. It separates the effects of logging-policy support, reward-model generalization, slate interactions, and estimator uncertainty.

The project rule is simple:

> Measure overlap, estimation error, and finite-estimate coverage separately — and preserve negative findings alongside successful results.

## At a Glance

- **Research question:** when do importance weighting, direct modeling, and doubly robust estimation succeed or fail for recommendation slates?
- **Current release:** `v1.0.0` — completed synthetic research software, not a production OPE validator or peer-reviewed result
- **Estimators:** IPS, SNIPS, DM, DR, Switch-DR, and slate-specific structured variants
- **Main pattern:** a reward model can predict logged slates well but generalize poorly to target-policy slates
- **Important exception:** SNIPS can outperform contextual DR under sparse exact-slate overlap, but may be undefined when no target slates match
- **Diagnostic finding:** the preferred risk-warning signal reverses between additive and contextual-pairwise reward mechanisms
- **Inference boundary:** M7.2 intervals condition on fixed alert selections; pooled results do not have joint-seed uncertainty intervals

Numerical source: [evidence ledger](docs/m70_evidence.md). Statistical scope: [release audit](docs/m81_release_audit.md).

## Why This Project Exists

A new recommendation policy is often evaluated using interactions logged under a different policy. For ordered slates, exact target-policy actions can be rare, and a seemingly accurate reward model may extrapolate poorly to the actions that matter.

OPEBenchLab asks:

> When does additional modeling improve off-policy estimates, and when does it create new failure modes?

The experiments vary sample size, logging exploration, slate support, and synthetic reward structure while keeping oracle policy values separate from observed logs.

## Key Findings

### Sparse support can favor simpler estimators

M6.2 replication results for the contextual-pairwise slate setting at exploration `1.0`:

| Requests | SNIPS RMSE (finite trials only) | Contextual DR RMSE | SNIPS finite coverage |
|---:|---:|---:|---:|
| 500 | 0.03620 | 0.09506 | 93.3% |
| 2,000 | 0.01777 | 0.03557 | 100% |
| 10,000 | 0.00735 | 0.00656 | 100% |

**Interpretation:** SNIPS performs better on the finite subset at smaller sample sizes, but its RMSE cannot be interpreted without coverage. At 10,000 requests, contextual DR is slightly better in this setting. These are simulator-specific comparisons, not universal estimator rankings.

### Logged-data fit is not target-policy generalization

In the M6.3 contextual-pairwise diagnostic at 500 requests and exploration `1.0`, target-policy prediction RMSE was **0.1697**, versus **0.0435** on logged slates. The corresponding contextual DR RMSE was **0.0975**. Under the matched additive mechanism, target-policy prediction RMSE was **0.0147** and DR RMSE was **0.0319**.

The DR error decomposes into direct-model error and its importance-weighted correction. This helps explain the observed failures in the controlled simulator; it does not prove that the same mechanism dominates in other environments.

### More complicated warning scores do not consistently win

M7.2's extended held-out validation compared warning scores at a matched **25% alert budget**. At 500 requests and exploration `1.0`:

| Reward mechanism | ESS warning precision | Logged-validation-RMSE warning precision |
|---|---:|---:|
| Contextual pairwise | 65.6% | 36.7% |
| Additive | 5.6% | 32.2% |

Across 4,320 unique within-mechanism held-out trials, pooled matched-budget precision was **28.43%** for ESS, **27.59%** for logged validation RMSE, and **26.57%** for their combined score. The pooled values are descriptive. The mechanism reversal is a warning against assuming that a single diagnostic transfers reliably.

See [M7.2 validation](docs/m72_validation.md) and [M7.3 statistical reporting](docs/m73_statistical_reporting.md) for the exact protocol and caveats.

## Benchmark

The benchmark develops from contextual bandits to ordered top-K slates, with exact logging propensities, synthetic oracle values, and controlled additive or contextual-pairwise reward mechanisms.

It studies:

- overlap, effective sample size (ESS), and zero-match events;
- IPS, SNIPS, DM, DR, Switch-DR, and structured slate estimators;
- cross-fitted ridge reward models and target-policy prediction error;
- replication, support stress, and reward-mechanism comparisons;
- retrospective risk signals, frozen calibration, and held-out warning ranking.

Oracle values are used to score estimators and retrospective diagnostics, **not** as estimator inputs or deployable warning labels.

## Research Evolution

> contextual bandits → ordered slates → reward-model diagnostics → calibrated warnings → statistical audit

| Phase | Research question | Main result | Disposition |
|---|---|---|---|
| M0–M3 | Can the OPE data contracts and baseline estimators be tested? | Synthetic bandit generator and baseline OPE methods | Retain |
| M4–M5.8 | What changes with learned models and slate interactions? | Exact slate support collapses; contextual pair features matter | Retain |
| M5.9–M5.12 | Does contextual DR or adaptive selection improve reliability? | SNIPS often remains competitive; zero-match fallback restores finite coverage at a cost | Retain with caveats |
| M6.0–M6.3 | Can failures be reproduced and explained? | Reproducible reports, support stress, and DR error decomposition | Retain |
| M6.4–M6.5 | Can observable diagnostics warn of large errors? | Warning precision is regime-dependent; calibration does not guarantee transfer | Retain as research diagnostics |
| M7.0–M7.3 | Do the evidence and statistical comparisons hold together? | Audited ledger, figures, extended held-out analysis, and manuscript appendix | Retain with inferential qualifications |
| M8.0–M8.1 | What is defensible at release? | Explicit statistical audit and bounded v1.0.0 research-software scope | Release |

Detailed milestone commands and protocols live in [`docs/`](docs/) and the [changelog](CHANGELOG.md).

## Negative and Inconclusive Results

OPEBenchLab retains results that argue *against* a universal improvement story:

- **Adaptive clipping and ESS-only selection (M5.10):** can choose badly under reward-model extrapolation error.
- **Conservative estimator selection (M5.11):** overwhelmingly selects SNIPS and does not establish a general-purpose DR advantage.
- **Support-aware fallback (M5.12):** produces finite estimates on zero-match trials, but does not preserve SNIPS's conditional RMSE advantage automatically.
- **Risk warning calibration (M6.5–M7.2):** combined warning scores do not consistently outperform ESS; transferring frozen thresholds between mechanisms can yield almost no alerts.

These are genuine negative or inconclusive findings in the studied simulator, not evidence that the corresponding methods always fail.

## Evaluation Protocol

The simulator separates observed logged outcomes from evaluation-only oracle truth. Modern experiments use reproducible seed schedules, paired mechanisms where specified, and independent calibration and held-out seeds for warning studies.

M7.2's matched-budget warning comparison ranks held-out **observable scores** within each configuration; it does not use held-out oracle errors to choose alerts. This is a *transductive ranking evaluation*, not a frozen deployment rule. The 12 held-out seed clusters support exploratory paired analysis, not thousands of independent inferential replicates.

The published seed-cluster bootstrap intervals resample **fixed original alert decisions**, without re-ranking within each resample. Shared seeds across configurations are also not modeled by a pooled joint-seed bootstrap. Both limitations remain open for a later paper; see the [M8.1 release audit](docs/m81_release_audit.md).

## Reproducibility

Install Python 3.11 or newer and the development dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

List and run a small modern benchmark:

```bash
python -m opebenchlab.m60_benchmark list
python -m opebenchlab.m60_benchmark run m512-support \
  --output-dir reports/m60_example --trials 1 --requests 500 \
  --explorations 0.2 --ridges 0.1
```

Inspect additional reproducibility and figure commands:

```bash
python -m opebenchlab.m62_robustness --help
python -m opebenchlab.m65_calibration --help
python -m opebenchlab.m72_validation --help
python -m opebenchlab.m73_manuscript --help
```

The M7.1 and M7.3 scripts generate research figures and SHA-256 manifests from archived experiment outputs. **Those original experiment CSV archives and rendered PNGs are not included in the Git source archive.** This README therefore does not embed missing images or imply that source checkout alone reproduces the historical figures. See [figure provenance](docs/m71_figures.md), [research reproducibility](docs/m70_reproducibility.md), and the [release checklist](docs/m81_release_checklist.md).

## Repository Structure

```text
OPEBenchLab/
├── docs/                # protocols, evidence, research draft, statistical audit
├── experiments/         # experiment helpers
├── src/opebenchlab/     # simulator, estimators, benchmark and analysis CLIs
├── tests/               # unit and regression tests
├── pyproject.toml       # Python package metadata
├── CHANGELOG.md         # release history
└── README.md
```

Generated `reports/` directories contain experiment results and are not part of the committed source archive.

## Experimental Discipline

- Keep oracle reward and policy-value information outside estimator fitting and warning decisions.
- Separate calibration from held-out evaluation; preserve paired trials and seed-level dependence.
- Report SNIPS finite-estimate coverage alongside conditional error.
- Distinguish retrospective oracle-labeled diagnostics from deployable warning rules.
- Record negative results, protocol qualifications, and provenance instead of silently discarding them.

## Limitations

- All main experiments use controlled synthetic environments from one simulator family, not production randomized recommendation logs.
- Exact-slate support can be extremely sparse; conditional SNIPS RMSE and always-defined DR RMSE answer different coverage questions.
- The 36 M7.2 paired intervals are unadjusted for multiple comparisons and are exploratory.
- M7.2 bootstrap intervals condition on fixed alert selections; pooled cross-configuration uncertainty does not account for shared seeds.
- Warning high-error labels require simulator truth, unavailable in ordinary deployment.
- Archived experiment CSVs must be distributed separately for independent numerical reproduction.

These constraints limit external validity and inferential strength; they do not prevent a reproducible **research software** release.

## What the Experiments Suggest

The current evidence supports a conditional, rather than universal, view of estimator reliability:

- **overlap** determines whether importance-weighted estimates are even defined and how noisy they become;
- **target-policy generalization** matters more than logged-data fit alone;
- **reward structure** changes which estimators and warning signals appear effective.

## Future Research

Potential follow-up studies — **not** unfinished requirements for v1.0.0 — include re-ranked seed-cluster bootstrap inference, joint-seed pooled intervals, multiplicity-aware testing, new reward and logging mechanisms, and evaluation on real randomized logs.

## Milestones

| Milestone | Deliverable |
|---|---|
| M0–M5 | Contextual-bandit and ordered-slate OPE framework |
| M6 | Reproducible robustness, diagnostics, and warning calibration |
| M7 | Evidence synthesis, figures, held-out validation, manuscript reporting |
| M8.0–M8.1 | Independent audit and v1.0.0 research-software release |

## Current Conclusion

OPEBenchLab asks:

> When can off-policy estimates of recommendation slates be trusted?

The controlled experiments suggest:

> **Not on estimator sophistication or logged-data fit alone.**

Support, target-policy generalization, and reward structure jointly determine which methods behave well. Simpler estimators can outperform more elaborate ones under sparse support, and warning signals can reverse their apparent usefulness across mechanisms. These are bounded synthetic findings, with statistical limitations explicitly documented rather than hidden.
