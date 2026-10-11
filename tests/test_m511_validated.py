import numpy as np
import pytest

from opebenchlab.m511_validated import M511Config, choose, estimate, run, summarize


def test_selector_requires_variance_gain_and_agreement():
    stats = dict(snips=0.5, contextual_dr=0.501, snips_se=0.1, contextual_dr_se=0.02)
    assert choose(stats) == 'contextual-dr'
    assert choose({**stats, 'contextual_dr': 1.0}) == 'snips'
    assert choose({**stats, 'contextual_dr_se': 0.09}) == 'snips'


def test_no_matches_falls_back_to_snips():
    stats = estimate(np.array([0.1, 0.2, 0.3]), np.zeros(3),
                     np.zeros(3), np.ones(3) * 0.4)
    assert np.isnan(stats['snips'])
    assert choose(stats) == 'snips'


def test_config_validation():
    with pytest.raises(ValueError):
        M511Config(train_fraction=0.8, validation_fraction=0.25)


def test_reproducible_large_request_regression():
    config = M511Config(trials=1, requests=(1202,), explorations=(0.2,), ridges=(0.1,))
    a, b = run(config), run(config)
    assert a == b
    assert len(a) == 4
    assert len(summarize(a)) == 4
    assert all(row['train_size'] + row['validation_size'] + row['test_size'] == 1202 for row in a)
