"""Monte Carlo reliability benchmark for single-action off-policy evaluation.

Oracle rewards are used exclusively for scoring, never by estimators or models.
Bootstrap intervals are descriptive percentile intervals, not guaranteed-valid CIs.
"""
from dataclasses import dataclass
from math import isfinite

import numpy as np

from opebenchlab.estimators import direct_method, doubly_robust, ips, snips, switch_dr
from opebenchlab.synthetic import (
    LoggedDataset, SyntheticConfig, SyntheticWorld, epsilon_greedy_probabilities,
    generate_requests, log_interactions, oracle_policy_value, uniform_probabilities,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_trials: int = 20
    n_requests: int = 1000
    exploration: float = 0.2
    seed: int = 42
    n_bootstrap: int = 100
    confidence: float = 0.95
    switch_threshold: float = 10.0

    def __post_init__(self):
        if self.n_trials < 2 or self.n_requests < 1 or self.n_bootstrap < 2:
            raise ValueError("need >=2 trials, >=1 request and >=2 bootstrap draws")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not isfinite(self.exploration) or not 0 < self.exploration <= 1:
            raise ValueError("exploration must be in (0, 1] for target-policy support")
        if not isfinite(self.switch_threshold) or self.switch_threshold < 0:
            raise ValueError("switch_threshold must be nonnegative and finite")


@dataclass(frozen=True)
class TrialResult:
    trial: int
    estimator: str
    estimate: float
    truth: float
    ess: float
    max_weight: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class ReliabilitySummary:
    estimator: str
    trials: int
    bias: float
    variance: float
    rmse: float
    mean_ess: float
    mean_max_weight: float
    ci_coverage: float


def _constant_reward_model(request, item_id):
    """Intentionally simplistic pre-action baseline; never uses oracle outcomes."""
    return 0.5


def _estimate(data, target, name, threshold):
    if name == "IPS":
        return ips(data, target)
    if name == "SNIPS":
        return snips(data, target)
    if name == "DM":
        return direct_method(data, target, _constant_reward_model)
    if name == "DR":
        return doubly_robust(data, target, _constant_reward_model)
    if name == "Switch-DR":
        return switch_dr(data, target, _constant_reward_model, threshold)
    raise ValueError(f"unknown estimator: {name}")


def _bootstrap_interval(data, target, name, threshold, rng, draws, confidence):
    n = len(data.events)
    values = []
    for _ in range(draws):
        indices = rng.integers(n, size=n)
        # Request IDs are duplicated in bootstrap samples; re-key them to keep
        # the estimator's alignment and uniqueness checks meaningful.
        from opebenchlab.schema import RecommendationRequest, LoggedBanditEvent
        requests = []
        events = []
        for j, i in enumerate(indices):
            req, ev = data.requests[int(i)], data.events[int(i)]
            key = f"bootstrap-{j}"
            requests.append(RecommendationRequest(key, req.user_id, req.candidate_item_ids, req.context))
            events.append(LoggedBanditEvent(key, ev.item_id, ev.logging_propensity, ev.reward, ev.logging_policy_id))
        sampled = LoggedDataset(tuple(requests), tuple(events))
        try:
            values.append(_estimate(sampled, target, name, threshold).value)
        except ValueError as exc:
            # SNIPS can be undefined in a resample containing no positive weights.
            if name != "SNIPS" or "no logged target-policy support" not in str(exc):
                raise
    if len(values) < 2:
        return float("nan"), float("nan")
    alpha = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(values, [alpha, 1 - alpha]))


def run_benchmark(config: BenchmarkConfig) -> tuple[TrialResult, ...]:
    """Vary logged samples and rewards; score against per-trial conditional truth.

    The environment seed stays fixed across trials, while request and logging
    seeds change. This evaluates conditional-on-request OPE error, not population
    generalization. The target is uniform and logging is epsilon-greedy.
    """
    results = []
    names = ("IPS", "SNIPS", "DM", "DR", "Switch-DR")
    for trial in range(config.n_trials):
        sim = SyntheticConfig(n_requests=config.n_requests, exploration=config.exploration,
                              seed=config.seed + trial * 1009)
        world = SyntheticWorld(sim)
        requests = generate_requests(sim)
        logging = lambda request: epsilon_greedy_probabilities(request, sim.exploration)
        data = log_interactions(world, requests, logging, seed=sim.seed + 2)
        truth = oracle_policy_value(world, requests, uniform_probabilities,
                                    f"trial-{trial}", "uniform").true_expected_reward
        rng = np.random.default_rng(sim.seed + 3)
        for name in names:
            estimate = _estimate(data, uniform_probabilities, name, config.switch_threshold)
            lo, hi = _bootstrap_interval(data, uniform_probabilities, name,
                                         config.switch_threshold, rng,
                                         config.n_bootstrap, config.confidence)
            results.append(TrialResult(trial, name, estimate.value, truth,
                                       estimate.effective_sample_size, estimate.max_weight,
                                       lo, hi))
    return tuple(results)


def summarize(results: tuple[TrialResult, ...]) -> tuple[ReliabilitySummary, ...]:
    if not results:
        raise ValueError("results must not be empty")
    summaries = []
    for name in dict.fromkeys(r.estimator for r in results):
        group = [r for r in results if r.estimator == name]
        errors = np.asarray([r.estimate - r.truth for r in group])
        coverage = [r.ci_lower <= r.truth <= r.ci_upper for r in group
                    if isfinite(r.ci_lower) and isfinite(r.ci_upper)]
        summaries.append(ReliabilitySummary(
            name, len(group), float(np.mean(errors)), float(np.var(errors, ddof=1))
            if len(errors) > 1 else float("nan"),
            float(np.sqrt(np.mean(errors ** 2))),
            float(np.mean([r.ess for r in group])),
            float(np.mean([r.max_weight for r in group])),
            float(np.mean(coverage)) if coverage else float("nan")))
    return tuple(summaries)
