import json
import numpy as np
from opebenchlab.m64_risk import Config, run, summarize, main


def test_small_run_has_observable_signals():
    rows = run(Config(trials=2, seeds=(42,), requests=(120,), explorations=(1.0,), ridges=(0.1, 1.0)))
    assert len(rows) == 4
    for row in rows:
        assert row['validation_rmse'] >= 0
        assert row['ridge_target_range'] >= 0
        assert row['split_target_disagreement'] >= 0
        assert row['validation_target_matches'] >= 0
        assert row['test_ess'] >= 0


def test_summary_counts_and_oracle_separation():
    rows = run(Config(trials=4, seeds=(42,), requests=(120,), explorations=(1.0,), ridges=(0.1,)))
    summary = summarize(rows)
    assert len(summary) == 10
    for row in summary:
        assert row['true_positives'] + row['false_positives'] == row['flagged']
        assert 0 <= row['precision'] <= 1


def test_cli_and_protocol(tmp_path):
    main(['--trials', '1', '--seeds', '42', '--requests', '120',
          '--explorations', '1', '--ridges', '0.1', '--output-dir', str(tmp_path)])
    assert (tmp_path/'diagnostics.csv').exists()
    assert (tmp_path/'risk_summary.csv').exists()
    protocol = json.loads((tmp_path/'protocol.json').read_text())
    assert 'dr_error' in protocol['oracle_only']
    assert 'dr_error' not in protocol['observable_signals']
