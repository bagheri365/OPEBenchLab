import numpy as np
import pytest

from opebenchlab.m63_diagnostics import Config, decompose, run, summarize


def test_decomposition_identity():
    p = decompose(np.array([.4,.6]), np.array([0.,2.]),
                  np.array([.1,.8]), np.array([.2,.5]), .55)
    assert p['dr_error'] == pytest.approx(p['dm_error']+p['correction'])
    assert p['dr_error_sq'] == pytest.approx(p['dm_error_sq']+p['correction_sq']+p['cross_term'])


def test_zero_match_is_finite_dr():
    p = decompose(np.array([.4,.6]), np.zeros(2),
                  np.array([.1,.8]), np.array([.2,.5]), .55)
    assert p['correction'] == 0
    assert np.isfinite(p['dr'])


def test_config_validation():
    with pytest.raises(ValueError):
        Config(seeds=(42,42))
    with pytest.raises(ValueError):
        Config(k=8)


def test_small_paired_run_and_summary():
    cfg = Config(trials=2, requests=(120,), explorations=(1.0,),
                 seeds=(42,), ridges=(0.1,))
    rows = run(cfg)
    assert len(rows) == 4
    assert {r['mechanism'] for r in rows} == {'additive','contextual-pairwise'}
    for r in rows:
        assert r['dr_error_sq'] == pytest.approx(r['dm_error_sq']+r['correction_sq']+r['cross_term'])
        assert r['target_prediction_rmse'] >= 0
    summary = summarize(rows)
    assert len(summary) == 2
    for s in summary:
        assert s['dr_mse'] == pytest.approx(s['dm_mse']+s['correction_second_moment']+s['cross_second_moment'])
