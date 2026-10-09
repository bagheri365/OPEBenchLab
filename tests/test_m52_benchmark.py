import math

import pytest

from opebenchlab.m52_benchmark import M52Config, run, summarize, main


@pytest.mark.parametrize('kwargs', [
    {'trials': 0}, {'requests': (0,)}, {'ks': (0,)}, {'ks': (6,)},
    {'explorations': (0,)}, {'explorations': (float('nan'),)},
    {'candidates': 0}, {'requests': ()},
])
def test_invalid_grid(kwargs):
    with pytest.raises(ValueError):
        M52Config(**kwargs)


def test_paired_rows_and_summary():
    rows = run(M52Config(trials=3, requests=(15,), ks=(2,), explorations=(0.05,)))
    assert len(rows) == 6
    assert {r['estimator'] for r in rows} == {'IPS', 'SNIPS'}
    for ips, snips in zip(rows[::2], rows[1::2]):
        assert ips['trial'] == snips['trial']
        assert ips['oracle'] == snips['oracle']
        assert ips['matches'] == snips['matches']
        assert ips['ess'] == snips['ess']
        assert 0 <= ips['matches'] <= 15
        assert 0 <= ips['ess'] <= 15
        assert ips['undefined'] == 0
        assert snips['undefined'] == (snips['matches'] == 0)
    summary = summarize(rows)
    assert len(summary) == 2
    assert summary[0]['trials'] == 3
    assert 0 <= summary[1]['undefined_frequency'] <= 1


def test_deterministic():
    config = M52Config(trials=2, requests=(12,), ks=(1, 3), explorations=(0.2,))
    first, second = run(config), run(config)
    for a, b in zip(first, second):
        assert a.keys() == b.keys()
        assert all(a[key] == b[key] or (isinstance(a[key], float) and math.isnan(a[key]) and math.isnan(b[key])) for key in a)


def test_all_undefined_summary():
    row = dict(requests=10, k=2, exploration=0.1, estimator='SNIPS', undefined=1,
               ess=0.0, max_weight=0.0, matches=0, match_rate=0.0, zero_match=1)
    result = summarize([row])[0]
    assert result['defined_trials'] == 0
    assert math.isnan(result['rmse'])
    assert result['undefined_frequency'] == 1


def test_cli_writes_files(tmp_path):
    main(['--trials', '1', '--requests', '12', '--ks', '1', '--explorations', '0.5', '--output-dir', str(tmp_path)])
    assert (tmp_path / 'summary.csv').exists()
    assert (tmp_path / 'trials.csv').exists()
