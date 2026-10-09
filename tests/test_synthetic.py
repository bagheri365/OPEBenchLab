import numpy as np
import pytest

from opebenchlab.synthetic import (
    SyntheticConfig, SyntheticWorld, epsilon_greedy_probabilities,
    generate_requests, log_interactions, make_dataset,
    oracle_policy_value, uniform_probabilities,
)


def config(**kwargs):
    return SyntheticConfig(**(dict(n_users=12, n_items=8, n_requests=80,
                                   candidates_per_request=4) | kwargs))


def test_reproducible():
    assert make_dataset(config(seed=19)) == make_dataset(config(seed=19))


def test_different_seeds_change_data():
    assert make_dataset(config(seed=19)) != make_dataset(config(seed=20))


def test_propensity_matches_policy():
    c = config(exploration=0.25)
    data = make_dataset(c)
    for request, event in zip(data.requests, data.events):
        probabilities = epsilon_greedy_probabilities(request, c.exploration)
        index = request.candidate_item_ids.index(event.item_id)
        assert event.logging_propensity == pytest.approx(probabilities[index])
        assert event.reward in (0.0, 1.0)


def test_policy_normalization_and_exploration():
    request = generate_requests(config())[0]
    probabilities = epsilon_greedy_probabilities(request, 0.3)
    assert sum(probabilities) == pytest.approx(1)
    assert np.count_nonzero(probabilities) == 4
    assert max(probabilities) == pytest.approx(0.7 + 0.3 / 4)


def test_oracle_isolated_from_logged_data():
    c = config()
    world = SyntheticWorld(c)
    requests = generate_requests(c)
    data = log_interactions(world, requests, uniform_probabilities)
    oracle = oracle_policy_value(world, requests, uniform_probabilities, "s1", "uniform")
    assert 0 < oracle.true_expected_reward < 1
    assert not hasattr(data, "oracle")
    assert all(not hasattr(event, "click_probability") for event in data.events)


def test_oracle_is_exact_conditional_expectation():
    c = config()
    world = SyntheticWorld(c)
    requests = generate_requests(c)
    value = oracle_policy_value(world, requests, uniform_probabilities, "s1", "uniform")
    manual = np.mean([np.mean([world.click_probability(req, item)
                               for item in req.candidate_item_ids]) for req in requests])
    assert value.true_expected_reward == pytest.approx(manual)


def test_invalid_config():
    with pytest.raises(ValueError):
        config(exploration=-0.1)
    with pytest.raises(ValueError):
        config(candidates_per_request=9)


def test_invalid_policy():
    c = config()
    world = SyntheticWorld(c)
    requests = generate_requests(c)
    with pytest.raises(ValueError, match="policy"):
        log_interactions(world, requests, lambda r: np.array([0.5, 0.5]))


def test_no_global_rng_dependency():
    c = config()
    np.random.seed(0)
    first = make_dataset(c)
    np.random.seed(12345)
    assert first == make_dataset(c)
