import numpy as np
import pytest

from opebenchlab.m59_dr import M59Config, run, summarize


def test_validation():
    with pytest.raises(ValueError):
        M59Config(explorations=(0,))
    with pytest.raises(ValueError):
        M59Config(folds=1)


def test_deterministic_and_finite():
    config = M59Config(trials=1, requests=(50,), explorations=(0.2,), ridges=(0.1, 1.0))
    first = run(config)
    assert first == run(config)
    assert len(first) == 8
    assert all(np.isfinite(row['estimate']) for row in first)
    assert all(row['eval_size'] == 40 for row in first)
    assert len(summarize(first)) == 8


def test_exact_propensity_and_ess():
    rows = run(M59Config(trials=1, requests=(60,), explorations=(1.0,), ridges=(1.0,)))
    ips = next(r for r in rows if r['estimator'] == 'ips')
    assert ips['max_weight'] in (0.0, 60.0)
    assert 0 <= ips['ess'] <= ips['eval_size']


def test_zero_noise_supported():
    rows = run(M59Config(trials=1, requests=(50,), explorations=(0.5,), noise=0))
    assert any(r['estimator'] == 'dr' and r['model'] == 'contextual' for r in rows)
