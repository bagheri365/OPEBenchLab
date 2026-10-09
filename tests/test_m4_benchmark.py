import math

import pytest

from opebenchlab.m4_benchmark import M4Config, run, summarize, write_csv


def test_paired_design_and_summary():
    rows = run(M4Config(trials=2, requests=30, explorations=(0.2,), folds=3))
    assert len(rows) == 8
    for trial in (0, 1):
        group = [r for r in rows if r['trial'] == trial]
        assert len({r['truth'] for r in group}) == 1
        ess_values = [r["ess"] for r in group]
        assert ess_values == pytest.approx([ess_values[0]] * len(ess_values))
        assert math.isnan(group[0]['brier'])
    summaries = summarize(rows)
    assert len(summaries) == 4
    assert all(s['trials'] == 2 and math.isfinite(s['rmse']) for s in summaries)


def test_reproducible_and_csv(tmp_path):
    config = M4Config(trials=2, requests=25, explorations=(0.5,), folds=2)
    a = run(config)
    b = run(config)
    for x, y in zip(a, b):
        for key in ('estimate', 'truth', 'error', 'ess'):
            assert x[key] == pytest.approx(y[key])
    write_csv(tmp_path / 'summary.csv', summarize(a))
    assert (tmp_path / 'summary.csv').read_text().startswith('exploration,estimator,')


@pytest.mark.parametrize('kwargs', [dict(trials=1), dict(requests=1),
                                   dict(folds=501), dict(explorations=(0.0,)),
                                   dict(ridge=0)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        M4Config(**kwargs)
