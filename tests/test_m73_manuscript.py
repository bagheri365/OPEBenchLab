import csv
import json
from pathlib import Path

import pytest

from opebenchlab.m73_manuscript import evidence, read_csv, main


def test_missing_columns(tmp_path):
    p = tmp_path/'x.csv'
    p.write_text('a\n1\n')
    with pytest.raises(ValueError, match='missing columns'):
        read_csv(p, ('b',))


def test_missing_mechanism():
    with pytest.raises(ValueError, match='missing signal'):
        evidence([{'transfer':'within','mode':'matched','mechanism':'additive','signal':'ess',
                   'flagged':'1','true_positives':'1','trials':'4'}], [])


def test_evidence_pooled_counts_and_pairing():
    rows = [dict(transfer='within',mode='matched',mechanism=m,signal=s,
                 flagged='2',true_positives=str(tp),trials='8')
            for m in ('additive','contextual_pairwise')
            for s,tp in (('ess',1),('validation',2),('combined',0))]
    paired = [dict(transfer='within',mechanism='contextual_pairwise',requests='500',
                   exploration='1.0',candidate=c,precision_difference='-0.2',
                   ci_low='-0.3',ci_high='-0.1',seeds='12')
              for c in ('validation','combined')]
    result, comparisons = evidence(rows,paired)
    assert len(result)==6
    assert next(r for r in result if r['mechanism']=='additive' and r['signal']=='validation')['precision']==1
    assert comparisons['combined']['seeds']==12


def test_cli_smoke(tmp_path):
    pytest.importorskip('matplotlib')
    src = tmp_path/'in'; src.mkdir()
    rows = [dict(transfer='within',mode='matched',mechanism=m,signal=s,
                 flagged=2,true_positives=tp,trials=8)
            for m in ('additive','contextual_pairwise')
            for s,tp in (('ess',1),('validation',2),('combined',0))]
    paired = [dict(transfer='within',mechanism='contextual_pairwise',requests=500,
                   exploration=1.0,candidate=c,precision_difference=-.2,
                   ci_low=-.3,ci_high=-.1,seeds=12)
              for c in ('validation','combined')]
    for name, data in [('warning_summary.csv',rows),('paired_precision.csv',paired)]:
        with (src/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    out=tmp_path/'out'
    main(['--input-dir',str(src),'--output-dir',str(out)])
    assert (out/'mechanism_reversal.png').stat().st_size>1000
    manifest=json.loads((out/'evidence_manifest.json').read_text())
    assert len(manifest['sources'])==2
    assert len(manifest['table'])==6
