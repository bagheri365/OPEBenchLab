import numpy as np
import pytest

from opebenchlab.m55_benchmark import M55Config, features, fit_ridge, run


def test_feature_encoding():
    candidates = ('i0', 'i1', 'i2', 'i3')
    slate = ('i2', 'i0', 'i3')
    additive = features(slate, candidates)
    pairwise = features(slate, candidates, True)
    assert additive.shape == (13,)
    assert pairwise.shape == (29,)
    assert additive.sum() == 4
    assert pairwise.sum() == 5
    assert pairwise[13 + 2 * 4 + 0] == 1


def test_invalid_inputs():
    with pytest.raises(ValueError):
        M55Config(train_fraction=1)
    with pytest.raises(ValueError):
        M55Config(ridge=0)
    with pytest.raises(ValueError):
        features(('i0', 'i0'), ('i0', 'i1'))
    with pytest.raises(ValueError):
        fit_ridge(np.ones((2, 2)), np.ones(3), 1)


def test_aggregate_benchmark_deterministic_and_paired():
    cfg = M55Config(trials=2, requests=(40,), ks=(3,), explorations=(0.5,), scenarios=('additive', 'pairwise', 'threeway'))
    rows = run(cfg)
    again = run(cfg)
    assert all(a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))
               for r1, r2 in zip(rows, again) for a, b in zip(r1.values(), r2.values()))
    assert len(rows) == 24
    for scenario in cfg.scenarios:
        for trial in range(2):
            group = [r for r in rows if r['scenario'] == scenario and r['trial'] == trial]
            assert len(group) == 4
            assert len({r['oracle'] for r in group}) == 1
            assert {r['train_requests'] for r in group} == {20}
            assert {r['eval_requests'] for r in group} == {20}
            assert all(r['undefined'] == int(not np.isfinite(r['estimate'])) for r in group)


def test_ridge_intercept_unpenalized():
    x = np.column_stack([np.ones(8), np.arange(8)])
    y = np.full(8, 0.7)
    beta = fit_ridge(x, y, 100)
    assert np.allclose(x @ beta, y)
