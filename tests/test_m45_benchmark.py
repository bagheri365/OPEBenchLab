import pytest
from opebenchlab.m45_benchmark import M45Config, run, paired_comparisons


def test_invalid_config():
    with pytest.raises(ValueError):
        M45Config(requests=(10,), folds=6)
    with pytest.raises(ValueError):
        M45Config(train_fraction=1.0)
    with pytest.raises(ValueError):
        M45Config(fixed_ridge=123)


def test_paired_evaluation_and_selection():
    rows, choices = run(M45Config(trials=2, requests=(30,), explorations=(0.2,),
                                   ridges=(1.0, 10.0), folds=3, capacity=4))
    assert len(rows) == 6
    assert len(choices) == 2
    for trial in (0, 1):
        group = [r for r in rows if r['trial'] == trial]
        assert {r['strategy'] for r in group} == {'fixed', 'independent-brier', 'evaluation-oof'}
        assert len({r['truth'] for r in group}) == 1
        assert [r['ess'] for r in group] == pytest.approx([group[0]['ess']] * 3)
        assert all(r['train_requests'] == 15 and r['eval_requests'] == 15 for r in group)
    assert len(paired_comparisons(rows)) == 2


def test_reproducible():
    config = M45Config(trials=2, requests=(30,), explorations=(0.2,),
                       ridges=(1.0, 10.0), folds=3, capacity=4)
    a, ca = run(config)
    b, cb = run(config)
    assert ca == cb
    for x, y in zip(a, b):
        assert x['estimate'] == pytest.approx(y['estimate'])
