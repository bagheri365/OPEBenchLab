import numpy as np
import pytest

from opebenchlab.m57_audit import M57Config, audit_design, projection_residual, run, summarize


def test_target_relative_signal_not_fixed_pair_function():
    _, slates, pairs, additive, generic, targeted, signal = audit_design(4, 3)
    assert len(slates) == 24 and len(pairs) == 12
    assert projection_residual(additive, signal) > 0.1
    assert projection_residual(generic, signal) > 0.1
    assert projection_residual(targeted, signal) < 1e-10


def test_exhaustive_rank_and_counts():
    _, slates, pairs, additive, generic, targeted, signal = audit_design(5, 3)
    assert len(slates) == 60 and len(pairs) == 20
    assert len(signal) == 1200
    assert np.linalg.matrix_rank(generic) < generic.shape[1]
    assert targeted.shape[1] == additive.shape[1] + 1


def test_deterministic_and_target_recovery():
    cfg = M57Config(trials=2, requests=(500,), ridges=(0.001,), candidates=4, noise=0)
    rows = run(cfg)
    assert rows == run(cfg)
    assert len(rows) == 6
    by_model = {r['model']: r for r in rows if r['trial'] == 0}
    assert abs(by_model['target-pair-diagnostic']['target_error']) < 0.005
    assert by_model['generic-pairwise']['representation_rmse'] > 0
    assert len(summarize(rows)) == 3


def test_invalid_config():
    with pytest.raises(ValueError):
        M57Config(k=8)
    with pytest.raises(ValueError):
        M57Config(noise=-1)
    with pytest.raises(ValueError):
        M57Config(ridges=(0,))
