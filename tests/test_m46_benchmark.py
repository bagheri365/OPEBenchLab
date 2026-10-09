import pytest
from opebenchlab.m46_benchmark import M46Config, run, paired_comparisons, main


def test_invalid_config():
    with pytest.raises(ValueError):
        M46Config(requests=(10,), folds=6)
    with pytest.raises(ValueError):
        M46Config(ridges=(0.1, 10.0))
    with pytest.raises(ValueError):
        M46Config(fixed_ridge=100)


def test_three_strategies_share_untouched_evaluation():
    cfg = M46Config(trials=2, requests=(30,), explorations=(0.2,),
                    ridges=(0.1, 10.0, 100.0), folds=3, capacity=4)
    rows, choices = run(cfg)
    assert len(rows) == 6 and len(choices) == 2
    for trial in range(2):
        group = [r for r in rows if r['trial'] == trial]
        assert {r['strategy'] for r in group} == {'fixed-10', 'fixed-100', 'independent-brier'}
        assert len({r['truth'] for r in group}) == 1
        assert [r['ess'] for r in group] == pytest.approx([group[0]['ess']] * 3)
        assert all(r['train_requests'] == 15 and r['eval_requests'] == 15 for r in group)
        selected = next(r for r in group if r['strategy'] == 'independent-brier')
        assert selected['selected_ridge'] == choices[trial]['brier_ridge']
    pairs = paired_comparisons(rows)
    assert {p['comparator'] for p in pairs} == {'fixed-10', 'fixed-100'}
    assert all(p['trials'] == 2 for p in pairs)


def test_reproducible():
    cfg = M46Config(trials=2, requests=(30,), explorations=(0.2,), folds=3, capacity=4)
    a, ac = run(cfg)
    b, bc = run(cfg)
    assert ac == bc
    assert [r['estimate'] for r in a] == pytest.approx([r['estimate'] for r in b])


def test_cli_writes_outputs(tmp_path):
    main(['--trials', '2', '--requests', '30', '--explorations', '0.2',
          '--folds', '3', '--capacity', '4', '--output-dir', str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} == {'summary.csv', 'trials.csv', 'paired.csv', 'choices.csv'}
