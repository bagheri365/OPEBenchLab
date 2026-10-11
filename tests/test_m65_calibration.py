import pytest
from opebenchlab.m65_calibration import evaluate, _metrics


def rows(mechanism, seeds, errors):
    return [dict(mechanism=mechanism, requests=500, exploration=1.,
                 seed=seed, trial=i, dr_error=err,
                 test_ess=float(i), validation_rmse=float(i))
            for seed in seeds for i, err in enumerate(errors)]


def test_seed_leakage_rejected():
    data = rows('additive', [1], [0,1,2,3])
    with pytest.raises(ValueError, match='disjoint'):
        evaluate(data, data)


def test_frozen_threshold_and_transfer():
    cal = rows('contextual-pairwise', [1,2], [0,1,2,3])
    test = rows('contextual-pairwise', [3,4], [0,0,0,10])
    test += rows('additive', [3,4], [0,0,0,0])
    frozen, predictions, summary = evaluate(cal, test, bootstrap=10)
    assert len(frozen) == 3
    assert len(predictions) == 48
    assert len(summary) == 6
    assert all(r['error_threshold']==2.25 for r in predictions)
    assert all(r['high_error']==0 for r in predictions if r['mechanism']=='additive')
    assert all(r['source_mechanism']=='contextual-pairwise' for r in predictions)


def test_no_oracle_signal_leakage():
    cal = rows('additive', [1,2], [0,1,2,3])
    a = rows('additive', [3,4], [0,0,0,10])
    b = rows('additive', [3,4], [10,10,10,10])
    _, pa, _ = evaluate(cal, a, bootstrap=0)
    _, pb, _ = evaluate(cal, b, bootstrap=0)
    assert [(r['risk_score'],r['warning']) for r in pa] == [(r['risk_score'],r['warning']) for r in pb]
    assert [r['high_error'] for r in pa] != [r['high_error'] for r in pb]


def test_empty_positive_metrics():
    m = _metrics([dict(warning=0,high_error=0), dict(warning=1,high_error=0)])
    assert m['recall'] != m['recall']
    assert m['false_positive_rate'] == .5
