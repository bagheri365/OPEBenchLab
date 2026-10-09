"""Reproducible single-action bandit simulation with a private reward oracle.

Requests and events contain only observable information. Never train an OPE
nuisance model on ``SyntheticWorld`` internals or ``oracle_policy_value``.
"""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Callable

import numpy as np

from opebenchlab.schema import (
    LoggedBanditEvent, OraclePolicyValue, RecommendationRequest,
    validate_logged_action,
)


@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 100
    n_items: int = 20
    n_requests: int = 1000
    candidates_per_request: int = 5
    embedding_dim: int = 4
    exploration: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.n_users, self.n_items, self.n_requests, self.embedding_dim) <= 0:
            raise ValueError("counts and embedding_dim must be positive")
        if not 1 <= self.candidates_per_request <= self.n_items:
            raise ValueError("candidates_per_request must be between 1 and n_items")
        if not isfinite(self.exploration) or not 0 <= self.exploration <= 1:
            raise ValueError("exploration must be in [0, 1]")


@dataclass(frozen=True)
class LoggedDataset:
    requests: tuple[RecommendationRequest, ...]
    events: tuple[LoggedBanditEvent, ...]


class SyntheticWorld:
    """Hidden reward mechanism; use only for simulation and benchmark scoring."""

    def __init__(self, config: SyntheticConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._users = rng.normal(size=(config.n_users, config.embedding_dim))
        self._items = rng.normal(size=(config.n_items, config.embedding_dim))
        self._quality = rng.normal(scale=0.5, size=config.n_items)

    def click_probability(self, request: RecommendationRequest, item_id: str) -> float:
        """Oracle-only conditional click probability; not a logged feature."""
        if item_id not in request.candidate_item_ids:
            raise ValueError("item is not eligible")
        user_idx = int(request.user_id[1:])
        item_idx = int(item_id[1:])
        if not 0 <= user_idx < self.config.n_users or not 0 <= item_idx < self.config.n_items:
            raise ValueError("unknown user or item")
        score = (float(self._users[user_idx] @ self._items[item_idx]) /
                 self.config.embedding_dim ** 0.5 + self._quality[item_idx]
                 + 0.3 * request.context[0])
        return 1.0 / (1.0 + exp(-score))


def epsilon_greedy_probabilities(
    request: RecommendationRequest, exploration: float
) -> np.ndarray:
    """Observable deterministic preference for lowest item ID plus uniform exploration."""
    if not isfinite(exploration) or not 0 <= exploration <= 1:
        raise ValueError("exploration must be in [0, 1]")
    count = len(request.candidate_item_ids)
    probabilities = np.full(count, exploration / count, dtype=float)
    preferred = min(range(count), key=lambda i: int(request.candidate_item_ids[i][1:]))
    probabilities[preferred] += 1.0 - exploration
    return probabilities


def uniform_probabilities(request: RecommendationRequest) -> np.ndarray:
    return np.full(len(request.candidate_item_ids), 1 / len(request.candidate_item_ids))


Policy = Callable[[RecommendationRequest], np.ndarray]


def _validated_probabilities(request: RecommendationRequest, policy: Policy) -> np.ndarray:
    probabilities = np.asarray(policy(request), dtype=float)
    if (probabilities.shape != (len(request.candidate_item_ids),)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("policy must return finite, nonnegative probabilities summing to one")
    return probabilities


def generate_requests(config: SyntheticConfig) -> tuple[RecommendationRequest, ...]:
    """Generate pre-action requests independently of the logging policy."""
    rng = np.random.default_rng(config.seed + 1)
    requests = []
    for index in range(config.n_requests):
        user = int(rng.integers(config.n_users))
        candidates = tuple(f"i{int(i)}" for i in rng.choice(
            config.n_items, size=config.candidates_per_request, replace=False))
        context = (float(rng.normal()), float(rng.integers(2)))
        requests.append(RecommendationRequest(f"r{index}", f"u{user}", candidates, context))
    return tuple(requests)


def log_interactions(
    world: SyntheticWorld, requests: tuple[RecommendationRequest, ...],
    policy: Policy, seed: int = 123,
    policy_id: str = "logging-v1",
) -> LoggedDataset:
    """Draw one action and Bernoulli click per request using a local RNG."""
    if not policy_id:
        raise ValueError("policy_id must be nonempty")
    rng = np.random.default_rng(seed)
    events = []
    for request in requests:
        probabilities = _validated_probabilities(request, policy)
        chosen = int(rng.choice(len(probabilities), p=probabilities))
        item_id = request.candidate_item_ids[chosen]
        propensity = float(probabilities[chosen])
        if propensity <= 0:
            raise ValueError("sampled action has zero propensity")
        reward = float(rng.random() < world.click_probability(request, item_id))
        event = LoggedBanditEvent(request.request_id, item_id, propensity, reward, policy_id)
        validate_logged_action(request, event)
        events.append(event)
    return LoggedDataset(requests, tuple(events))


def oracle_policy_value(
    world: SyntheticWorld, requests: tuple[RecommendationRequest, ...],
    target_policy: Policy, scenario_id: str, target_policy_id: str,
) -> OraclePolicyValue:
    """Exact expected reward conditional on these requests (not population value)."""
    if not requests:
        raise ValueError("requests must be nonempty")
    total = 0.0
    for request in requests:
        probabilities = _validated_probabilities(request, target_policy)
        total += sum(float(p) * world.click_probability(request, item)
                     for p, item in zip(probabilities, request.candidate_item_ids))
    return OraclePolicyValue(scenario_id, target_policy_id, total / len(requests))


def make_dataset(config: SyntheticConfig) -> LoggedDataset:
    world = SyntheticWorld(config)
    requests = generate_requests(config)
    return log_interactions(
        world, requests,
        lambda request: epsilon_greedy_probabilities(request, config.exploration),
        seed=config.seed + 2,
        policy_id="epsilon-greedy-v1",
    )
