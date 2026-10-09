"""M5.1: exact ordered-slate logging and full-slate IPS/SNIPS.

Sequential weighted sampling without replacement defines a Plackett-Luce
policy. The logged propensity is the product of *conditional* probabilities.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, isfinite

import numpy as np

from opebenchlab.synthetic import SyntheticConfig, SyntheticWorld, generate_requests
from opebenchlab.schema import RecommendationRequest


@dataclass(frozen=True)
class SlateEvent:
    request_id: str
    slate: tuple[str, ...]
    conditional_propensities: tuple[float, ...]
    logging_propensity: float
    reward: float


@dataclass(frozen=True)
class SlateDataset:
    requests: tuple[RecommendationRequest, ...]
    events: tuple[SlateEvent, ...]


@dataclass(frozen=True)
class SlateEstimate:
    value: float
    effective_sample_size: float
    max_weight: float
    n_events: int


def _scores(request: RecommendationRequest, scores: np.ndarray) -> np.ndarray:
    result = np.asarray(scores, dtype=float)
    if result.shape != (len(request.candidate_item_ids),) or not np.all(np.isfinite(result)) or np.any(result < 0) or not np.any(result > 0):
        raise ValueError('scores must be finite, nonnegative, nonzero and match candidates')
    return result


def slate_probability(request: RecommendationRequest, slate: tuple[str, ...], scores: np.ndarray) -> tuple[float, ...]:
    """Conditional probabilities of an ordered slate; zeros indicate no support."""
    weights = _scores(request, scores).copy()
    if not 1 <= len(slate) <= len(weights) or len(set(slate)) != len(slate):
        raise ValueError('slate must contain distinct eligible items and valid K')
    conditionals = []
    for item in slate:
        if item not in request.candidate_item_ids:
            raise ValueError('slate contains ineligible item')
        index = request.candidate_item_ids.index(item)
        total = float(weights.sum())
        conditional = float(weights[index] / total) if total > 0 else 0.0
        conditionals.append(conditional)
        weights[index] = 0.0
    return tuple(conditionals)


def sample_slate(request: RecommendationRequest, scores: np.ndarray, k: int, rng: np.random.Generator) -> tuple[tuple[str, ...], tuple[float, ...]]:
    weights = _scores(request, scores).copy()
    if not 1 <= k <= len(weights) or np.count_nonzero(weights) < k:
        raise ValueError('K exceeds available positive-probability candidates')
    chosen, conditionals = [], []
    for _ in range(k):
        probabilities = weights / weights.sum()
        index = int(rng.choice(len(weights), p=probabilities))
        chosen.append(request.candidate_item_ids[index])
        conditionals.append(float(probabilities[index]))
        weights[index] = 0.0
    return tuple(chosen), tuple(conditionals)


def deterministic_top_k(request: RecommendationRequest, k: int) -> tuple[str, ...]:
    if not 1 <= k <= len(request.candidate_item_ids):
        raise ValueError('invalid K')
    return tuple(sorted(request.candidate_item_ids, key=lambda item: int(item[1:]))[:k])


def logging_scores(request: RecommendationRequest, exploration: float) -> np.ndarray:
    if not isfinite(exploration) or not 0 < exploration <= 1:
        raise ValueError('exploration must be in (0, 1] to ensure full support')
    scores = np.full(len(request.candidate_item_ids), exploration, dtype=float)
    preferred = request.candidate_item_ids.index(min(request.candidate_item_ids, key=lambda item: int(item[1:])))
    scores[preferred] += 1 - exploration
    return scores


def expected_slate_reward(world: SyntheticWorld, request: RecommendationRequest, slate: tuple[str, ...]) -> float:
    """Oracle-only expected mean position-weighted clicks, normalized by K."""
    if not slate or len(set(slate)) != len(slate) or any(item not in request.candidate_item_ids for item in slate):
        raise ValueError('invalid slate')
    return float(sum(world.click_probability(request, item) / (position + 1) for position, item in enumerate(slate)) / len(slate))


def log_slates(config: SyntheticConfig, k: int) -> tuple[SlateDataset, float]:
    if not 1 <= k <= config.candidates_per_request:
        raise ValueError('invalid K')
    if config.exploration <= 0:
        raise ValueError('exploration must be positive for full-slate support')
    world = SyntheticWorld(config)
    requests = generate_requests(config)
    rng = np.random.default_rng(config.seed + 517)
    events = []
    oracle_values = []
    for request in requests:
        slate, conditional = sample_slate(request, logging_scores(request, config.exploration), k, rng)
        clicks = [float(rng.random() < world.click_probability(request, item) / (position + 1)) for position, item in enumerate(slate)]
        events.append(SlateEvent(request.request_id, slate, conditional, float(np.prod(conditional)), float(np.mean(clicks))))
        oracle_values.append(expected_slate_reward(world, request, deterministic_top_k(request, k)))
    return SlateDataset(requests, tuple(events)), float(np.mean(oracle_values))


def slate_weights(data: SlateDataset, k: int) -> np.ndarray:
    if not data.requests or len(data.requests) != len(data.events) or len({r.request_id for r in data.requests}) != len(data.requests):
        raise ValueError('nonempty aligned requests and events with unique IDs required')
    weights = []
    for request, event in zip(data.requests, data.events):
        if request.request_id != event.request_id or len(event.slate) != k:
            raise ValueError('misaligned request or incorrect slate length')
        expected = slate_probability(request, event.slate, logging_scores(request, 1.0))  # validate slate eligibility
        if len(expected) != k or len(event.conditional_propensities) != k:
            raise ValueError('invalid conditional propensities')
        if any(not isfinite(p) or p <= 0 or p > 1 for p in event.conditional_propensities):
            raise ValueError('invalid conditional propensities')
        if not isfinite(event.logging_propensity) or event.logging_propensity <= 0 or not np.isclose(event.logging_propensity, np.prod(event.conditional_propensities), rtol=1e-10, atol=1e-14):
            raise ValueError('joint propensity must equal conditional product')
        if not isfinite(event.reward) or not 0 <= event.reward <= 1:
            raise ValueError('reward must be in [0, 1]')
        weights.append((1.0 if event.slate == deterministic_top_k(request, k) else 0.0) / event.logging_propensity)
    return np.asarray(weights, dtype=float)


def _estimate(data: SlateDataset, k: int, normalized: bool) -> SlateEstimate:
    weights = slate_weights(data, k)
    rewards = np.asarray([event.reward for event in data.events], dtype=float)
    total = float(weights.sum())
    if normalized and total == 0:
        raise ValueError('SNIPS undefined: no exact target slate observed')
    value = float(np.dot(weights, rewards) / (total if normalized else len(weights)))
    squared = float(np.dot(weights, weights))
    return SlateEstimate(value, total ** 2 / squared if squared else 0.0, float(weights.max()), len(weights))


def slate_ips(data: SlateDataset, k: int) -> SlateEstimate:
    return _estimate(data, k, normalized=False)


def slate_snips(data: SlateDataset, k: int) -> SlateEstimate:
    return _estimate(data, k, normalized=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='M5.1 exact full-slate IPS/SNIPS demonstration')
    parser.add_argument('--requests', type=int, default=10000)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--exploration', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = SyntheticConfig(n_requests=args.requests, candidates_per_request=args.candidates, exploration=args.exploration, seed=args.seed)
    data, oracle = log_slates(config, args.k)
    ips = slate_ips(data, args.k)
    print(f'Oracle: {oracle:.6f} | IPS: {ips.value:.6f} | ESS: {ips.effective_sample_size:.1f}/{ips.n_events} | max weight: {ips.max_weight:.2f}')
    try:
        snips = slate_snips(data, args.k)
        print(f'SNIPS: {snips.value:.6f}')
    except ValueError as error:
        print(f'SNIPS unavailable: {error}')


if __name__ == '__main__':
    main()
