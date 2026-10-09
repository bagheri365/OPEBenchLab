import math

import pytest

from opebenchlab.benchmark import BenchmarkConfig, run_benchmark, summarize


def test_invalid_config():
    for kwargs in ({"n_trials": 1}, {"n_bootstrap": 1}, {"exploration": 0},
                   {"confidence": 1}, {"switch_threshold": -1}):
        with pytest.raises(ValueError):
            BenchmarkConfig(**kwargs)


def test_reproducibility_and_estimators():
    config = BenchmarkConfig(n_trials=2, n_requests=100, n_bootstrap=5, seed=11)
    first = run_benchmark(config)
    second = run_benchmark(config)
    assert len(first) == len(second)
    for actual, repeated in zip(first, second):
        assert (actual.trial, actual.estimator) == (repeated.trial, repeated.estimator)
        for field in ("estimate", "truth", "ess", "max_weight", "ci_lower", "ci_upper"):
            assert getattr(actual, field) == pytest.approx(
                getattr(repeated, field), rel=1e-12, abs=1e-12
            )
    assert len(first) == 10
    assert {r.estimator for r in first} == {"IPS", "SNIPS", "DM", "DR", "Switch-DR"}
    assert all(0 <= r.truth <= 1 for r in first)


def test_summary_matches_errors():
    results = run_benchmark(BenchmarkConfig(n_trials=3, n_requests=100,
                                             n_bootstrap=5, seed=7))
    summaries = summarize(results)
    assert len(summaries) == 5
    assert all(s.trials == 3 and s.rmse >= 0 and s.mean_ess >= 0 for s in summaries)
    assert all(0 <= s.ci_coverage <= 1 or math.isnan(s.ci_coverage) for s in summaries)
    dm = next(s for s in summaries if s.estimator == "DM")
    assert dm.variance >= 0


def test_empty_summary():
    with pytest.raises(ValueError):
        summarize(())
