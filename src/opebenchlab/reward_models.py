"""Outcome-only nuisance models for OPE; no synthetic oracle access.

The feature map is deliberately simple/misspecified. All inputs are available
before action selection; logged rewards are used only for fitting. Cross-fitting
keeps each evaluation event's reward out of its own nuisance model training.
"""
from dataclasses import dataclass

import numpy as np

from opebenchlab.schema import RecommendationRequest
from opebenchlab.synthetic import LoggedDataset


def pre_action_features(request: RecommendationRequest, item_id: str) -> np.ndarray:
    """Public feature map; never use reward, propensity or oracle values."""
    if item_id not in request.candidate_item_ids:
        raise ValueError("item must be eligible")
    user = int(request.user_id[1:])
    item = int(item_id[1:])
    context = np.asarray(request.context, dtype=float)
    if len(context) < 1:
        raise ValueError("context must contain at least one value")
    # Numeric IDs are intentionally weak proxy features, not oracle embeddings.
    return np.array([1.0, np.sin(user), np.cos(item), context[0],
                     np.sin(user) * np.cos(item), context[0] * np.cos(item)], dtype=float)


@dataclass(frozen=True)
class LinearRewardModel:
    coefficients: np.ndarray

    def __call__(self, request: RecommendationRequest, item_id: str) -> float:
        prediction = float(pre_action_features(request, item_id) @ self.coefficients)
        return float(np.clip(prediction, 0.0, 1.0))


def fit_reward_model(data: LoggedDataset, *, ridge: float = 1.0) -> LinearRewardModel:
    """Fit a ridge least-squares probability predictor to observed clicks only."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    if not data.events or len(data.events) != len(data.requests):
        raise ValueError("nonempty aligned requests and events required")
    if len({r.request_id for r in data.requests}) != len(data.requests):
        raise ValueError("request IDs must be unique")
    for request, event in zip(data.requests, data.events):
        if request.request_id != event.request_id or event.item_id not in request.candidate_item_ids:
            raise ValueError("invalid logged event alignment")
    x = np.stack([pre_action_features(r, e.item_id) for r, e in zip(data.requests, data.events)])
    y = np.asarray([e.reward for e in data.events], dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return LinearRewardModel(coef)


@dataclass(frozen=True)
class CrossFittedRewardModel:
    """Request-indexed out-of-fold predictors; reject unseen requests."""
    models: tuple[LinearRewardModel, ...]
    request_to_fold: dict[str, int]

    def __call__(self, request: RecommendationRequest, item_id: str) -> float:
        if request.request_id not in self.request_to_fold:
            raise ValueError("cross-fitted model only supports evaluation requests")
        return self.models[self.request_to_fold[request.request_id]](request, item_id)


def cross_fit_reward_model(data: LoggedDataset, *, folds: int = 3,
                           seed: int = 42, ridge: float = 1.0) -> CrossFittedRewardModel:
    """Split by request, train on other folds, predict all candidates OOF."""
    n = len(data.requests)
    if not 2 <= folds <= n:
        raise ValueError("folds must be between 2 and number of requests")
    if len({r.request_id for r in data.requests}) != n:
        raise ValueError("request IDs must be unique")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    models = []
    assignment = {}
    for fold_id, heldout in enumerate(np.array_split(permutation, folds)):
        train_mask = np.ones(n, dtype=bool)
        train_mask[heldout] = False
        training = LoggedDataset(tuple(r for i, r in enumerate(data.requests) if train_mask[i]),
                                 tuple(e for i, e in enumerate(data.events) if train_mask[i]))
        models.append(fit_reward_model(training, ridge=ridge))
        for idx in heldout:
            assignment[data.requests[int(idx)].request_id] = fold_id
    return CrossFittedRewardModel(tuple(models), assignment)


def brier_score(data: LoggedDataset, model) -> float:
    """Observed-action predictive error; not counterfactual policy-value error."""
    if not data.events:
        raise ValueError("data must be nonempty")
    return float(np.mean([(model(r, e.item_id) - e.reward) ** 2
                          for r, e in zip(data.requests, data.events)]))
