import pytest

from opebenchlab.schema import (
    LoggedBanditEvent,
    OraclePolicyValue,
    RecommendationRequest,
    validate_logged_action,
)


def request():
    return RecommendationRequest("r1", "u1", ("a", "b"), (0.1, 1.0))


def event(**overrides):
    values = dict(request_id="r1", item_id="a", logging_propensity=0.25,
                  reward=1.0, logging_policy_id="epsilon-greedy-v1")
    values.update(overrides)
    return LoggedBanditEvent(**values)


def test_valid_logged_action():
    validate_logged_action(request(), event())


@pytest.mark.parametrize("propensity", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_propensity(propensity):
    with pytest.raises(ValueError, match="propensity"):
        event(logging_propensity=propensity)


def test_logged_action_must_be_eligible():
    with pytest.raises(ValueError, match="candidate set"):
        validate_logged_action(request(), event(item_id="missing"))


def test_request_linkage():
    with pytest.raises(ValueError, match="request_id mismatch"):
        validate_logged_action(request(), event(request_id="r2"))


def test_duplicate_candidates_rejected():
    with pytest.raises(ValueError, match="unique"):
        RecommendationRequest("r1", "u1", ("a", "a"), ())


def test_oracle_is_distinct_contract():
    oracle = OraclePolicyValue("baseline", "target-v1", 0.42)
    assert oracle.true_expected_reward == 0.42
    assert not hasattr(event(), "true_expected_reward")
