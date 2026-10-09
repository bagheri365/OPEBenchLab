"""Minimal M0 contracts for single-action contextual-bandit evaluation.

All request features must be known *before* the logging policy chooses an action.
Oracle values are evaluation-only and must never enter logged feedback or training.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RecommendationRequest:
    request_id: str
    user_id: str
    candidate_item_ids: tuple[str, ...]
    context: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.request_id or not self.user_id:
            raise ValueError("request_id and user_id must be nonempty")
        if not self.candidate_item_ids or len(set(self.candidate_item_ids)) != len(self.candidate_item_ids):
            raise ValueError("candidates must be nonempty and unique")
        if not all(isfinite(x) for x in self.context):
            raise ValueError("context values must be finite")


@dataclass(frozen=True)
class LoggedBanditEvent:
    request_id: str
    item_id: str
    logging_propensity: float
    reward: float
    logging_policy_id: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.item_id or not self.logging_policy_id:
            raise ValueError("identifiers must be nonempty")
        if not isfinite(self.logging_propensity) or not 0 < self.logging_propensity <= 1:
            raise ValueError("logging propensity must be in (0, 1]")
        if not isfinite(self.reward):
            raise ValueError("reward must be finite")


@dataclass(frozen=True)
class OraclePolicyValue:
    """Ground truth for benchmark scoring only, not an estimator input."""

    scenario_id: str
    target_policy_id: str
    true_expected_reward: float

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.target_policy_id:
            raise ValueError("identifiers must be nonempty")
        if not isfinite(self.true_expected_reward):
            raise ValueError("true expected reward must be finite")


def validate_logged_action(request: RecommendationRequest, event: LoggedBanditEvent) -> None:
    """Check event linkage and action eligibility, without consulting oracle data."""
    if request.request_id != event.request_id:
        raise ValueError("request_id mismatch")
    if event.item_id not in request.candidate_item_ids:
        raise ValueError("logged item is not in the request candidate set")
