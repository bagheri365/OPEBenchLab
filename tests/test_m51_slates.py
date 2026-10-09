import itertools

import numpy as np
import pytest

from opebenchlab.m51_slates import (
    SlateDataset, SlateEvent, deterministic_top_k, log_slates, logging_scores,
    sample_slate, slate_ips, slate_probability, slate_snips, slate_weights,
)
from opebenchlab.synthetic import SyntheticConfig, generate_requests


def test_ordered_slate_probabilities_normalize():
    request = generate_requests(SyntheticConfig(n_requests=1, candidates_per_request=4))[0]
    scores = np.array([1., 2., 3., 4.])
    for k in (1, 2, 3, 4):
        total = sum(np.prod(slate_probability(request, slate, scores)) for slate in itertools.permutations(request.candidate_item_ids, k))
        assert total == pytest.approx(1)


def test_conditional_not_original_marginal_product():
    request = generate_requests(SyntheticConfig(n_requests=1, candidates_per_request=3))[0]
    scores = np.array([1., 2., 3.])
    slate = request.candidate_item_ids[:2]
    p = slate_probability(request, slate, scores)
    assert p == pytest.approx((1 / 6, 2 / 5))
    assert np.prod(p) != pytest.approx((1 / 6) * (2 / 6))


def test_reproducibility_and_oracle():
    config = SyntheticConfig(n_requests=80, exploration=0.5)
    a, oracle_a = log_slates(config, 2)
    b, oracle_b = log_slates(config, 2)
    assert a == b and oracle_a == oracle_b
    assert 0 <= oracle_a <= 1
    assert all(len(set(e.slate)) == 2 for e in a.events)
    assert all(e.logging_propensity == pytest.approx(np.prod(e.conditional_propensities)) for e in a.events)


def test_ips_snips_hand_calculation():
    request = generate_requests(SyntheticConfig(n_requests=1))[0]
    slate = deterministic_top_k(request, 2)
    event = SlateEvent(request.request_id, slate, (0.5, 0.5), 0.25, 0.75)
    data = SlateDataset((request,), (event,))
    assert slate_weights(data, 2).tolist() == [4.0]
    assert slate_ips(data, 2).value == pytest.approx(3.0)
    assert slate_snips(data, 2).value == pytest.approx(0.75)
    assert slate_ips(data, 2).effective_sample_size == pytest.approx(1)


def test_no_target_match_snips_undefined():
    request = generate_requests(SyntheticConfig(n_requests=1))[0]
    slate = tuple(reversed(deterministic_top_k(request, 2)))
    data = SlateDataset((request,), (SlateEvent(request.request_id, slate, (0.5, 0.5), 0.25, 1.),))
    assert slate_ips(data, 2).value == 0
    with pytest.raises(ValueError, match='undefined'):
        slate_snips(data, 2)


def test_invalid_inputs():
    request = generate_requests(SyntheticConfig(n_requests=1))[0]
    with pytest.raises(ValueError):
        slate_probability(request, (request.candidate_item_ids[0],) * 2, np.ones(len(request.candidate_item_ids)))
    with pytest.raises(ValueError):
        log_slates(SyntheticConfig(exploration=0), 2)
    with pytest.raises(ValueError):
        sample_slate(request, np.ones(len(request.candidate_item_ids)), len(request.candidate_item_ids) + 1, np.random.default_rng(1))
    with pytest.raises(ValueError):
        logging_scores(request, 0)
