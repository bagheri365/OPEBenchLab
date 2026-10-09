import numpy as np
import pytest

from opebenchlab.m43_benchmark import M43Config, run, summarize_m43, paired_m43, main


def test_shapes_pairing_and_external_baseline():
    rows = run(M43Config(trials=2, requests=(20,), explorations=(0.2,),
                         ridges=(0.1, 10.0), capacity=4, folds=2))
    assert len(rows) == 12
    for trial in (0, 1):
        group = [r for r in rows if r['trial'] == trial]
        assert len({r['truth'] for r in group}) == 1
        assert np.allclose([r['ess'] for r in group], group[0]['ess'])
        assert {r['strategy'] for r in group} == {'in-sample', 'cross-fitted', 'external'}
    assert len(summarize_m43(rows)) == 6
    assert len(paired_m43(rows)) == 2


def test_reproducibility():
    config = M43Config(trials=2, requests=(20,), explorations=(0.2,),
                       ridges=(1.0,), capacity=3, folds=2)
    a, b = run(config), run(config)
    assert [r['estimate'] for r in a] == pytest.approx([r['estimate'] for r in b])


@pytest.mark.parametrize('changes', [
    {'trials': 1}, {'requests': (2,), 'folds': 3},
    {'explorations': (0,)}, {'ridges': (0,)}, {'capacity': -1},
])
def test_bad_config(changes):
    with pytest.raises(ValueError):
        M43Config(**changes)


def test_cli_writes_files(tmp_path):
    main(['--trials', '2', '--requests', '20', '--explorations', '0.2',
          '--ridges', '1', '--capacity', '3', '--folds', '2',
          '--output-dir', str(tmp_path)])
    assert all((tmp_path / f).is_file() for f in ('trials.csv', 'summary.csv', 'paired.csv'))
