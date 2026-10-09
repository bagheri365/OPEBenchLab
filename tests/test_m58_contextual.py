import numpy as np
import pytest

from opebenchlab.m58_contextual import (M58Config, contextual_indices,
                                        fit_context_residual, run, summarize)


def test_contextual_encoding_distinguishes_context():
    pairs = [('a', 'b'), ('b', 'a')]
    ix = contextual_indices([('a', 'b'), ('a', 'b')], pairs, pairs)
    assert ix.tolist() == [0, 2]


def test_shrinkage_and_empty_cells():
    result = fit_context_residual(np.array([0, 0, 1]),
                                  np.array([1., 3., 2.]), 2., 3)
    np.testing.assert_allclose(result, [1., 2./3., 0.])


def test_invalid_config():
    with pytest.raises(ValueError):
        M58Config(exploration=0)
    with pytest.raises(ValueError):
        M58Config(ridges=(0,))


def test_run_deterministic_and_summary():
    config = M58Config(trials=1, requests=(50,), ridges=(0.1,), noise=0.0)
    rows = run(config)
    assert len(rows) == 4
    assert rows == run(config)
    assert {r['model'] for r in rows} == {'additive', 'generic-pairwise',
                                        'target-pair-oracle', 'contextual-pairwise'}
    assert len(summarize(rows)) == 4
    assert all(np.isfinite(r['target_error']) for r in rows)
