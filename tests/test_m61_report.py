import csv
import hashlib
import json
import math

import pytest

from opebenchlab.m61_report import analyze, main, verify_manifest


def test_paired_excludes_nonfinite_and_reports_coverage():
    rows = []
    for trial, snips, dr in [(0, 1., 0.), (1, float('nan'), 2.), (2, 2., 1.)]:
        for name, error in [('snips', snips), ('contextual-dr', dr)]:
            rows.append(dict(requests='500', exploration='1.0', trial=str(trial), estimator=name, error=str(error)))
    coverage, paired = analyze(rows, bootstrap=100, seed=7)
    assert next(r for r in coverage if r['estimator'] == 'snips')['coverage'] == pytest.approx(2/3)
    assert next(r for r in coverage if r['estimator'] == 'contextual-dr')['coverage'] == 1
    assert paired[0]['paired_trials'] == 2
    assert paired[0]['excluded_nonfinite_or_missing'] == 1
    assert paired[0]['mean_dr_minus_snips_squared_error'] == -2
    assert paired[0]['bootstrap_ci_upper'] <= -1


def test_empty_paired_group_and_duplicate_rejected():
    row = dict(requests='1', exploration='1', trial='0', estimator='snips', error='nan')
    coverage, paired = analyze([row], bootstrap=100)
    assert coverage[0]['coverage'] == 0
    assert math.isnan(paired[0]['mean_dr_minus_snips_squared_error'])
    with pytest.raises(ValueError, match='Duplicate'):
        analyze([row, row], bootstrap=100)


def test_manifest_checksum_and_end_to_end(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'trials.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['requests','exploration','trial','estimator','error'])
        writer.writeheader()
        for name, error in [('snips', 0.2), ('contextual-dr', 0.1)]:
            writer.writerow(dict(requests=500, exploration=1, trial=0, estimator=name, error=error))
    manifest = {'experiment':'m512-support', 'artifacts': {'trials.csv': {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'columns': ['requests','exploration','trial','estimator','error'], 'rows':2}}}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    assert verify_manifest(source) == manifest
    main(['--input-dir',str(source),'--output-dir',str(tmp_path/'report'),'--bootstrap','100'])
    assert (tmp_path/'report'/'paired_bootstrap.csv').exists()
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        verify_manifest(source)


def test_invalid_bootstrap_and_columns():
    with pytest.raises(ValueError, match='bootstrap'):
        analyze([], bootstrap=10)
    with pytest.raises(ValueError, match='Trials CSV'):
        analyze([{'wrong': 1}])
