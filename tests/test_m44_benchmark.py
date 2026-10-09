import pytest

from opebenchlab.m44_benchmark import M44Config, choose_by_brier, paired, run, summarize


def test_config_validation():
    with pytest.raises(ValueError):
        M44Config(ridges=(0.1, 0.1))
    with pytest.raises(ValueError):
        M44Config(fixed_ridge=5)
    with pytest.raises(ValueError):
        M44Config(requests=(3,), folds=5)


def test_selection_only_uses_brier_and_ties_are_deterministic():
    assert choose_by_brier([(100, 0.2), (0.1, 0.1), (10, 0.3)]) == 0.1
    assert choose_by_brier([(10, 0.2), (0.1, 0.2)]) == 0.1


def test_paired_trials_and_oracle_reference():
    config = M44Config(trials=2, requests=(30,), explorations=(0.2,),
                       ridges=(0.1, 10.0), capacity=3, folds=3)
    rows, choices = run(config)
    assert len(rows) == 6
    assert len(choices) == 2
    assert len(summarize(rows)) == 3
    assert len(paired(rows)) == 2
    for trial in (0, 1):
        group = [r for r in rows if r['trial'] == trial]
        assert len({r['truth'] for r in group}) == 1
        assert [r['ess'] for r in group] == pytest.approx([group[0]['ess']] * 3)
        oracle = next(r for r in group if r['strategy'] == 'oracle-reference')
        assert oracle['squared_error'] <= min(r['squared_error'] for r in group) + 1e-12


def test_reproducible():
    config = M44Config(trials=2, requests=(30,), explorations=(0.2,),
                       ridges=(0.1, 10.0), capacity=3, folds=3)
    first, choices = run(config)
    second, second_choices = run(config)
    assert choices == second_choices
    for a, b in zip(first, second):
        assert a['strategy'] == b['strategy']
        assert a['estimate'] == pytest.approx(b['estimate'])
