import numpy as np
import pytest

from opebenchlab.estimators import (
    direct_method, doubly_robust, ips, snips, switch_dr,
)
from opebenchlab.schema import LoggedBanditEvent, RecommendationRequest
from opebenchlab.synthetic import LoggedDataset, uniform_probabilities


def example():
    requests = (
        RecommendationRequest('r0', 'u0', ('i0', 'i1'), (0.0,)),
        RecommendationRequest('r1', 'u1', ('i0', 'i1'), (1.0,)),
    )
    events = (
        LoggedBanditEvent('r0', 'i0', 0.25, 1.0, 'logger'),
        LoggedBanditEvent('r1', 'i1', 0.5, 0.0, 'logger'),
    )
    return LoggedDataset(requests, events)


def constant_model(request, item):
    return 0.25


def test_hand_computed_ips_snips_and_ess():
    data = example()
    assert ips(data, uniform_probabilities).value == pytest.approx(1.0)
    assert snips(data, uniform_probabilities).value == pytest.approx(2 / 3)
    assert ips(data, uniform_probabilities).effective_sample_size == pytest.approx(9 / 5)
    assert ips(data, uniform_probabilities).max_weight == pytest.approx(2)


def test_hand_computed_dm_dr_and_switch():
    data = example()
    policy = uniform_probabilities
    assert direct_method(data, policy, constant_model).value == pytest.approx(.25)
    assert doubly_robust(data, policy, constant_model).value == pytest.approx(.875)
    assert switch_dr(data, policy, constant_model, 1).value == pytest.approx(.125)
    assert switch_dr(data, policy, constant_model, 2).value == pytest.approx(.875)
    assert switch_dr(data, policy, constant_model, 0).value == pytest.approx(.25)


def test_snips_undefined_for_zero_overlap():
    data = example()
    def target(req):
        return np.array([1., 0.]) if req.request_id == 'r1' else np.array([0., 1.])
    assert ips(data, target).value == 0
    with pytest.raises(ValueError, match='SNIPS undefined'):
        snips(data, target)


def test_invalid_target_policy_and_model():
    data = example()
    with pytest.raises(ValueError, match='target policy'):
        ips(data, lambda req: np.array([.4, .4]))
    with pytest.raises(ValueError, match='reward model'):
        direct_method(data, uniform_probabilities, lambda req, item: float('nan'))
    with pytest.raises(ValueError, match='threshold'):
        switch_dr(data, uniform_probabilities, constant_model, -1)


def test_misaligned_events_rejected():
    data = example()
    bad = LoggedDataset(data.requests, data.events[::-1])
    with pytest.raises(ValueError, match='request_id mismatch'):
        ips(bad, uniform_probabilities)


def test_dr_matches_ips_for_zero_model():
    data = example()
    zero = lambda req, item: 0.0
    assert doubly_robust(data, uniform_probabilities, zero).value == ips(data, uniform_probabilities).value
