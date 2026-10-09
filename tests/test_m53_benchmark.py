import numpy as np
import pytest

from opebenchlab.m53_benchmark import M53Config, position_marginals, run, summarize_m53


def test_uniform_marginals():
    p = position_marginals(np.ones(5), 4)
    assert p.shape == (4, 5)
    assert np.allclose(p, 0.2)
    assert np.allclose(p.sum(axis=1), 1)


def test_two_item_marginals():
    p = position_marginals(np.array([3., 1.]), 2)
    assert np.allclose(p, [[.75, .25], [.25, .75]])


def test_marginal_input_validation():
    with pytest.raises(ValueError):
        position_marginals(np.array([1., 0.]), 1)
    with pytest.raises(ValueError):
        position_marginals(np.ones(3), 4)


def test_config_validation():
    with pytest.raises(ValueError):
        M53Config(ks=(1,))
    with pytest.raises(ValueError):
        M53Config(scenarios=('unknown',))


def test_run_is_reproducible_and_paired():
    cfg = M53Config(trials=2, requests=(30,), ks=(2,), explorations=(0.5,))
    a = run(cfg)
    assert a == run(cfg)
    assert len(a) == 16
    for scenario in ('additive', 'interaction'):
        group = [r for r in a if r['scenario'] == scenario]
        assert {r['estimator'] for r in group} == {'Slate IPS', 'Slate SNIPS', 'Position IPS', 'Position SNIPS'}
    assert len(summarize_m53(a)) == 8


def test_interaction_oracle_shift():
    rows = run(M53Config(trials=1, requests=(50,), ks=(2,), explorations=(1.0,), interaction_bonus=0.2))
    oracle = {r['scenario']: r['oracle'] for r in rows}
    assert oracle['interaction'] - oracle['additive'] == pytest.approx(0.2)
