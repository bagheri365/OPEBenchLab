import numpy as np
import pytest

from opebenchlab.estimators import direct_method, doubly_robust
from opebenchlab.reward_models import (
    brier_score, cross_fit_reward_model, fit_reward_model, pre_action_features,
)
from opebenchlab.synthetic import SyntheticConfig, make_dataset, uniform_probabilities


def sample():
    return make_dataset(SyntheticConfig(n_requests=120, seed=27))


def test_learned_models_integrate_with_dm_and_dr():
    data = sample()
    fitted = fit_reward_model(data)
    oof = cross_fit_reward_model(data, folds=3, seed=4)
    for model in (fitted, oof):
        assert 0 <= brier_score(data, model) <= 1
        assert np.isfinite(direct_method(data, uniform_probabilities, model).value)
        assert np.isfinite(doubly_robust(data, uniform_probabilities, model).value)


def test_crossfit_reproducible():
    data = sample()
    a = cross_fit_reward_model(data, folds=4, seed=12)
    b = cross_fit_reward_model(data, folds=4, seed=12)
    assert a.request_to_fold == b.request_to_fold
    for r in data.requests:
        for item in r.candidate_item_ids:
            assert a(r, item) == pytest.approx(b(r, item))


def test_oof_does_not_use_own_reward():
    data = sample()
    original = cross_fit_reward_model(data, folds=3, seed=7)
    from opebenchlab.schema import LoggedBanditEvent
    from opebenchlab.synthetic import LoggedDataset
    e = data.events[0]
    changed = LoggedDataset(data.requests, (LoggedBanditEvent(e.request_id, e.item_id,
        e.logging_propensity, 1.0 - e.reward, e.logging_policy_id),) + data.events[1:])
    refit = cross_fit_reward_model(changed, folds=3, seed=7)
    r = data.requests[0]
    assert original(r, r.candidate_item_ids[0]) == pytest.approx(refit(r, r.candidate_item_ids[0]))


def test_invalid_inputs_and_unseen_request():
    data = sample()
    with pytest.raises(ValueError):
        cross_fit_reward_model(data, folds=1)
    with pytest.raises(ValueError):
        fit_reward_model(data, ridge=0)
    with pytest.raises(ValueError):
        pre_action_features(data.requests[0], "not-a-candidate")
    oof = cross_fit_reward_model(data)
    from opebenchlab.schema import RecommendationRequest
    r = data.requests[0]
    unseen = RecommendationRequest("unseen", r.user_id, r.candidate_item_ids, r.context)
    with pytest.raises(ValueError):
        oof(unseen, unseen.candidate_item_ids[0])
