import numpy as np
import pytest
from opebenchlab.m511_validated import M511Config
from opebenchlab.m512_support import M512Config, support_regime, run, summarize, paired_comparison


def test_regimes_and_config():
    assert support_regime(0, 10) == 'zero-match'
    assert support_regime(2, 10) == 'low-ess'
    assert support_regime(10, 10) == 'adequate'
    with pytest.raises(ValueError):
        M512Config(low_ess_threshold=-1)


def test_coverage_preserves_undefined():
    rows = [dict(requests=500, exploration=1., estimator='snips', error=float('nan'),
                 test_zero_match=True, test_support_regime='zero-match', fallback_used=False),
            dict(requests=500, exploration=1., estimator='snips', error=0.1,
                 test_zero_match=False, test_support_regime='low-ess', fallback_used=False)]
    s = summarize(rows)[0]
    assert s['coverage'] == 0.5 and s['undefined_trials'] == 1
    assert np.isclose(s['conditional_rmse'], 0.1)


def test_paired_only_finite_intersection():
    rows = []
    for t, a, b in [(0, float('nan'), 0.1), (1, 0.2, 0.1)]:
        rows += [dict(requests=500, exploration=1., trial=t, estimator='snips', error=a),
                 dict(requests=500, exploration=1., trial=t, estimator='contextual-dr', error=b)]
    r = paired_comparison(rows)[0]
    assert r['paired_trials'] == 1 and r['excluded_nonfinite'] == 1
    assert r['mean_dr_minus_snips_squared_error'] < 0


def test_large_request_and_action_only_fallback():
    cfg = M512Config(M511Config(trials=1, requests=(1202,), explorations=(1.,), ridges=(0.1,)))
    rows = run(cfg)
    assert len(rows) == 5
    assert len(summarize(rows)) == 5
    assert len(paired_comparison(rows)) == 1
    by_name = {r['estimator']: r for r in rows}
    chosen = by_name['support-aware']
    assert chosen['support_selection_source'] == 'evaluation-actions-only'
    assert chosen['fallback_used'] == chosen['test_zero_match']
    expected = by_name['contextual-dr'] if chosen['test_zero_match'] else by_name['snips']
    assert chosen['estimate'] == expected['estimate']


def test_zero_match_fallback_uses_dr_without_rewards_for_selection(monkeypatch):
    import opebenchlab.m512_support as mod
    common = dict(requests=500, exploration=1., trial=0, test_ess=0., true_value=0.49)
    synthetic = [dict(common, estimator=name, estimate=value)
                 for name, value in [('snips', float('nan')), ('contextual-dr', 0.48),
                                     ('contextual-dm', 0.47), ('selected', float('nan'))]]
    monkeypatch.setattr(mod, 'run_m511', lambda config: synthetic)
    result = {r['estimator']: r for r in mod.run(M512Config())}
    assert result['support-aware']['estimate'] == 0.48
    assert result['support-aware']['fallback_used']
    assert not result['snips']['finite']
    assert result['support-aware']['finite']
