import numpy as np
import pytest

from opebenchlab.m54_benchmark import M54Config, pair_marginals, run
from opebenchlab.m53_benchmark import position_marginals


def test_pair_marginals_consistent_with_positions():
    scores = np.array([1., 2., 3., 4.])
    joint = pair_marginals(scores, 3)
    pos = position_marginals(scores, 3)
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.isclose(joint[i, j].sum(), 1)
            assert np.allclose(joint[i, j].sum(axis=1), pos[i])
            assert np.allclose(joint[i, j].sum(axis=0), pos[j])
            assert np.allclose(np.diag(joint[i, j]), 0)


def test_invalid_config():
    with pytest.raises(ValueError):
        M54Config(ks=(2,))
    with pytest.raises(ValueError):
        M54Config(scenarios=('unknown',))


def test_paired_and_reproducible():
    cfg = M54Config(trials=2, requests=(40,), ks=(3,), explorations=(0.5,), candidates=4)
    rows = run(cfg)
    assert rows == run(cfg)
    assert len(rows) == 2 * 3 * 6
    for trial in range(2):
        group = [r for r in rows if r['trial'] == trial]
        assert len({r['oracle'] - (0.2 if r['scenario'] == 'pairwise' else 0.4 if r['scenario'] == 'threeway' else 0) for r in group}) <= 3
        assert all(r['pair_matches'] >= r['full_matches'] for r in group)


def test_pairwise_component_recovers_pair_bonus_when_matched():
    cfg = M54Config(trials=1, requests=(200,), ks=(3,), explorations=(1.0,), candidates=4)
    rows = run(cfg)
    d = {(r['scenario'], r['estimator']): r for r in rows}
    assert d['pairwise', 'Pairwise IPS']['estimate'] >= d['additive', 'Position IPS']['estimate']
    assert np.isclose(d['threeway', 'Pairwise IPS']['estimate'], d['pairwise', 'Pairwise IPS']['estimate'])
