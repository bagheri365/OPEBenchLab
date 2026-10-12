# Slate Off-Policy Evaluation Under Support and Reward-Model Shift: A Synthetic Benchmark Study

**Status:** v1.0.0 accompanying research draft; not peer-reviewed or externally validated. The open M8.0 inference issues are documented in `m81_release_audit.md`.

## Abstract

We study off-policy evaluation (OPE) for ordered recommendation slates in a synthetic environment with known logging propensities and a separately accessible reward oracle. Full-slate importance weighting faces rapidly declining exact-slate support. Reward-model approaches can exploit structural assumptions but may generalize poorly from logged actions to target-policy slates. In a contextual-pairwise reward setting with 500 requests and exploration 1.0, the target-action prediction RMSE is approximately 0.170 versus 0.044 on logged actions. The direct-model bias is approximately −0.162; the mean doubly robust (DR) correction is +0.137, leaving residual DR bias of approximately −0.025. Follow-up risk-warning experiments show that low effective sample size can be informative in the hardest regimes, but an equal-weight combination with logged validation RMSE does not improve pooled held-out precision. In a larger independent-seed follow-up (12 held-out seeds), matched-budget precision is 28.43% for ESS, 27.59% for validation RMSE, and 26.57% for their combination across within-mechanism settings. These findings identify limitations of overlap-only and logged-validation-only reliability diagnostics. They are synthetic, mechanism-dependent results rather than a general estimator ranking.

## 1. Introduction

A target slate policy may differ from the policy that generated logged recommendations. OPE attempts to estimate target value without online deployment. Ordered top-K actions create a combinatorial support problem, while reward models must predict outcomes under a target action distribution not necessarily represented well in the log. OPEBenchLab provides controlled experiments to study these tensions. Our focus is failure characterization rather than declaring a universally best estimator.

## 2. Experimental setting

The repository implements a synthetic contextual recommendation environment, exact logging propensities, target policies, and a separately evaluated reward oracle. Its estimator lineage includes IPS, self-normalized IPS (SNIPS), direct method (DM), doubly robust (DR), clipped and switching variants, and structured slate estimators. Later milestones contrast additive rewards with contextual pairwise interactions. The experiment code and individual milestone protocols, rather than this narrative, define the precise data-generating mechanisms, randomization, and splits. The oracle is not used to fit the OPE estimators.

## 3. Methods and audit principles

**Support:** exact-slate match counts and effective sample size summarize the limited observed support for target-policy actions. **Model generalization:** logged-action and target-action prediction RMSE are compared using synthetic oracle labels solely for evaluation. **DR decomposition:** the direct-model error and weighted residual correction are analyzed with their second moments and cross term; their covariance alone is not the cross term. **Risk warnings:** M6.4 ranks observable warning signals retrospectively; M6.5 freezes thresholds using calibration seeds and evaluates them on disjoint held-out seeds. Within-mechanism generalization is distinct from pairwise-to-additive transfer. The latter reuses held-out additive trials and changes the prevalence of the source-defined high-error label.

## 4. Results

### 4.1 Support and reward structure

Earlier slate experiments document exact-match support collapse as slate size and candidate count increase. Position-wise and pairwise structural estimators can trade modeling assumptions for lower effective variance; they do not automatically handle arbitrary context-dependent interactions. The recovered M6.2 outputs verify SNIPS finite-estimate coverage falling from 100% to 40% as slate support is stressed (5 candidates, K=2 versus 7 candidates, K=3; 500 requests, exploration 1.0). Support-aware fallback remains finite but incurs greater error in the lowest-support configuration.

### 4.2 DR bias and residual correction

For 500 requests, exploration 1.0, and contextual-pairwise rewards, M6.3 records DM bias −0.1619, mean correction +0.1371, and DR bias −0.0247. The corresponding additive setting has DM bias +0.0032. The contextual-pairwise target prediction RMSE is 0.1697 versus 0.0435 on logged actions, while the additive model is approximately equally accurate on both distributions. This association is consistent with target-distribution extrapolation error; it does not, by itself, prove a unique causal mechanism.

### 4.3 Risk warnings

M6.4 retrospective precision is close to the 25% within-cell high-error prevalence for most signals, with important regime-specific exceptions for ESS. In M6.5 held-out evaluation, pooled within-mechanism warning precision is approximately 25.9% for low ESS, 25.4% for validation RMSE, and 23.9% for their combined percentile score. The combined warning does not establish improved discrimination. Under contextual-pairwise-to-additive transfer, validation-based thresholds can cease to trigger alerts, illustrating calibration shift.

M7.2 extended replication evaluates 4,320 unique within-mechanism held-out trials (12 seeds), with 1,080 alerts per warning at a matched 25% budget. Pooled precision is 28.43% for ESS, 27.59% for validation RMSE, and 26.57% for the combined score. At n=500, exploration 1.0, the contextual-pairwise precision difference (combined minus ESS) is −23.33 percentage points (seed-cluster bootstrap 95% interval [−34.59, −11.64] pp); validation minus ESS is −28.89 pp [−41.11, −17.78]. Under additive rewards at the same setting, ESS precision is 5.6% while validation reaches 32.2%; under contextual pairwise rewards, ESS reaches 65.6% versus validation at 36.7%. These are regime-specific contrasts, not proof of a universal ranking.

See [the numerical evidence ledger](m70_evidence.md) for audited values and denominators.

## 5. Discussion

Support and reward-model generalization interact: a model may fit logged observations yet predict target slates poorly, and exact-slate residual corrections may be too sparse to repair bias reliably. Support-aware fallback can improve finite-estimate coverage without necessarily improving squared-error risk. More complex warning scores can dilute rather than enhance a useful support signal. The strongest practical lesson is to report support, coverage, model assumptions, and oracle-separated evaluation together, rather than reporting only conditional RMSE among defined estimates.

## 6. Limitations

All findings come from synthetic mechanisms, not real-world randomized recommendation logs. Some reported SNIPS RMSE values condition on the estimator being defined, creating a coverage-versus-accuracy tradeoff. The target prediction RMSE and actual OPE error are oracle-only diagnostics. M6.4 thresholds were chosen retrospectively; M6.5 warning and outcome cutoffs are simulation-calibrated and do not yield universal deployment thresholds. M6.5 initially used three held-out seeds; M7.2 extended replication uses twelve independent held-out seeds, which improves but does not eliminate uncertainty. Frozen-threshold alert fractions differ across rules; M7.2 additionally compares rankings at matched alert budgets. Its ranking analysis is retrospective and not a prospective deployment rule. The 36 paired intervals are not multiplicity-adjusted. Moreover, these seed-cluster intervals condition on the original matched-budget alert rankings; they do not repeat ranking inside each bootstrap draw. Pooled precision values are descriptive, with no joint-seed interval accounting for dependence across mechanisms and configurations. The M6.1 and M6.2 output archives were recovered and re-audited in M7.1; source files and figure generation are described in `m71_figures.md`. Their bootstrap comparisons remain unadjusted for multiple testing, and M6.2 does not supply usable interval-coverage evidence.

## 7. Conclusion

The benchmark documents why support collapse, target-distribution model error, and nonportable risk calibration complicate slate OPE. It supports careful, mechanism-specific diagnosis rather than a universally superior estimator or a deployable reliability alarm. Next steps for external validity are additional independent seeds, new misspecification families, interval coverage studies, and randomized production-log evaluations where feasible.
