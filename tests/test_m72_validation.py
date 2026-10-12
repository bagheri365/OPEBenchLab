import pytest
from opebenchlab.m72_validation import _ranked_decisions, analyze


def sample():
    out = []
    for signal in ('ess','validation','combined'):
        for seed in (10,11,12):
            for trial in range(4):
                out.append(dict(source_mechanism='additive', mechanism='additive',
                    transfer='within', requests='500', exploration='1.0',
                    seed=seed, trial=trial, signal=signal,
                    risk_score=float(trial if signal != 'ess' else 3-trial),
                    warning=int(trial>2), high_error=int(trial==0)))
    return out


def test_exact_budget_and_pairing():
    summary, paired, decisions = analyze(sample(), budget=.25, bootstrap=30)
    assert len(summary) == 6
    assert len(paired) == 2
    assert len(decisions) == 72
    assert all(r['flagged'] == 3 for r in summary if r['mode']=='matched')
    assert next(r for r in paired if r['candidate']=='validation')['precision_difference'] == -1


def test_oracle_labels_cannot_change_warnings():
    original = sample()
    changed = [dict(r, high_error=1-r['high_error']) for r in original]
    a = analyze(original, bootstrap=0)[2]
    b = analyze(changed, bootstrap=0)[2]
    assert [(r['mode'],r['signal'],r['warning']) for r in a] == [(r['mode'],r['signal'],r['warning']) for r in b]


def test_unpaired_rejected():
    with pytest.raises(ValueError, match='unpaired'):
        analyze(sample()[:-1], bootstrap=0)


def test_tie_break_deterministic():
    rows = [dict(seed=1,trial=i,risk_score=1.) for i in (2,0,1,3)]
    assert _ranked_decisions(rows, .5) == [0,1,1,0]
