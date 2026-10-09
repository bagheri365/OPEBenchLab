import numpy as np
import pytest

from opebenchlab.m56_diagnostics import M56Config, coefficient_recovery, run, summarize, targeted_features


def test_noise_free_coefficient_recovery():
    coefficient, rank, columns = coefficient_recovery()
    assert abs(coefficient - 0.2) < 0.001
    assert rank <= columns


def test_target_pair_feature():
    candidates = tuple(f'item-{i}' for i in range(5))
    target = candidates[:3]
    assert targeted_features(target, target, candidates)[-1] == 1
    assert targeted_features((candidates[1], candidates[0], candidates[2]), target, candidates)[-1] == 0


def test_invalid_config():
    with pytest.raises(ValueError):
        M56Config(ridges=(0,))
    with pytest.raises(ValueError):
        M56Config(k=6)


def test_diagnostic_paired_and_deterministic():
    config = M56Config(trials=1, requests=(30,), explorations=(0.5,), ridges=(0.1, 10.0))
    rows = run(config)
    assert all((a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))) for r, s in zip(rows, run(config)) for a, b in zip(r.values(), s.values()))
    assert len(rows) == 6
    assert {r['model'] for r in rows} == {'additive', 'pairwise', 'target-pair'}
    assert all(r['train_requests'] + r['eval_requests'] == 30 for r in rows)
    assert len(summarize(rows)) == 6
    assert all(np.isfinite(r['interaction_effect']) for r in rows)
