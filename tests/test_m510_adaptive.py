import numpy as np
import pytest

from opebenchlab.m510_adaptive import M510Config, choose, run, summarize
from opebenchlab.m59_dr import M59Config, run as run_m59


def test_config_validation():
    with pytest.raises(ValueError):
        M510Config(clips=(0,))


def test_selector_does_not_use_ground_truth():
    assert choose({'snips': 0.1}, 100, 500) == 'snips'
    assert choose({'snips': 0.1}, 1, 500) == 'contextual-dm'
    assert choose({'snips': float('nan')}, 100, 500) == 'contextual-dm'


def test_paired_clips_and_summary():
    rows = run(M510Config(trials=1, requests=(50,), explorations=(1.0,),
                          ridges=(1,), clips=(2, 60)))
    names = {r['estimator'] for r in rows}
    assert 'contextual-dr-clip-2' in names
    assert 'contextual-dr-clip-60' in names
    assert 'contextual-dr-unclipped' in names
    assert 'snips' in names
    assert sum(r['selected'] for r in rows) == 1
    assert len(summarize(rows)) == len(names)


def test_large_request_regression():
    # Regression: M5.9 originally indexed the 1200-row slate feature matrix
    # with request indices above 1200.
    rows = run_m59(M59Config(trials=1, requests=(1202,), explorations=(0.2,), ridges=(1,)))
    assert len(rows) == 8
    assert all(np.isfinite(r['estimate']) for r in rows)
