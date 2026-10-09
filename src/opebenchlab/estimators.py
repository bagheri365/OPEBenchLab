"""Single-action off-policy estimators; no oracle access or model fitting.

Inputs are logged requests/events, a target policy and optionally a reward
predictor trained independently of evaluation outcomes (e.g. cross-fitting).
"""

from dataclasses import dataclass
from math import isfinite
from typing import Callable

import numpy as np

from opebenchlab.schema import RecommendationRequest, validate_logged_action
from opebenchlab.synthetic import LoggedDataset, Policy

RewardModel = Callable[[RecommendationRequest, str], float]


@dataclass(frozen=True)
class OPEEstimate:
    value: float
    effective_sample_size: float
    max_weight: float
    n_events: int


def _inputs(data: LoggedDataset, target_policy: Policy):
    if not data.events or len(data.requests) != len(data.events):
        raise ValueError("nonempty aligned requests and events required")
    if len({r.request_id for r in data.requests}) != len(data.requests):
        raise ValueError("request IDs must be unique")
    weights = []
    probabilities = []
    for request, event in zip(data.requests, data.events):
        validate_logged_action(request, event)
        p = np.asarray(target_policy(request), dtype=float)
        if (p.shape != (len(request.candidate_item_ids),)
                or not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-10)):
            raise ValueError("invalid target policy probabilities")
        action_index = request.candidate_item_ids.index(event.item_id)
        weights.append(float(p[action_index] / event.logging_propensity))
        probabilities.append(p)
    return np.asarray(weights), probabilities


def _result(value: float, weights: np.ndarray) -> OPEEstimate:
    weight_sum = float(weights.sum())
    weight_sq_sum = float(np.dot(weights, weights))
    ess = weight_sum ** 2 / weight_sq_sum if weight_sq_sum else 0.0
    return OPEEstimate(float(value), ess, float(weights.max()), len(weights))


def ips(data: LoggedDataset, target_policy: Policy) -> OPEEstimate:
    weights, _ = _inputs(data, target_policy)
    rewards = np.asarray([event.reward for event in data.events])
    return _result(np.mean(weights * rewards), weights)


def snips(data: LoggedDataset, target_policy: Policy) -> OPEEstimate:
    weights, _ = _inputs(data, target_policy)
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError("SNIPS undefined: no logged target-policy support")
    rewards = np.asarray([event.reward for event in data.events])
    return _result(np.dot(weights, rewards) / denominator, weights)


def _model_terms(data: LoggedDataset, probabilities: list[np.ndarray], model: RewardModel):
    baselines = []
    chosen_predictions = []
    for request, event, p in zip(data.requests, data.events, probabilities):
        predictions = np.asarray([model(request, item) for item in request.candidate_item_ids], dtype=float)
        if predictions.shape != p.shape or not np.all(np.isfinite(predictions)):
            raise ValueError("reward model must return finite predictions")
        baselines.append(float(np.dot(p, predictions)))
        chosen_predictions.append(float(predictions[request.candidate_item_ids.index(event.item_id)]))
    return np.asarray(baselines), np.asarray(chosen_predictions)


def direct_method(data: LoggedDataset, target_policy: Policy, model: RewardModel) -> OPEEstimate:
    weights, probabilities = _inputs(data, target_policy)
    baselines, _ = _model_terms(data, probabilities, model)
    return _result(np.mean(baselines), weights)


def doubly_robust(data: LoggedDataset, target_policy: Policy, model: RewardModel) -> OPEEstimate:
    weights, probabilities = _inputs(data, target_policy)
    baselines, chosen = _model_terms(data, probabilities, model)
    rewards = np.asarray([event.reward for event in data.events])
    return _result(np.mean(baselines + weights * (rewards - chosen)), weights)


def switch_dr(data: LoggedDataset, target_policy: Policy, model: RewardModel,
              threshold: float) -> OPEEstimate:
    """Use DR correction only for importance weights <= threshold.

    This is a variance-reduction heuristic and can introduce bias.
    """
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    weights, probabilities = _inputs(data, target_policy)
    baselines, chosen = _model_terms(data, probabilities, model)
    rewards = np.asarray([event.reward for event in data.events])
    corrections = np.where(weights <= threshold, weights * (rewards - chosen), 0.0)
    return _result(np.mean(baselines + corrections), weights)
