"""M7.3: auditable manuscript table and mechanism-reversal figure from M7.2 CSVs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path, required):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(set(required)-set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: empty data')
    return rows


def evidence(summary, paired):
    within = [r for r in summary if r['transfer']=='within' and r['mode']=='matched']
    if not within:
        raise ValueError('no within-mechanism matched-budget rows')
    mechanisms = ('additive', 'contextual_pairwise')
    output = []
    for mechanism in mechanisms:
        subset = [r for r in within if r['mechanism'].replace('-', '_') == mechanism]
        if not subset:
            raise ValueError(f'missing mechanism {mechanism}')
        for signal in ('ess', 'validation', 'combined'):
            selected = [r for r in subset if r['signal']==signal]
            if not selected:
                raise ValueError(f'missing signal {signal} for {mechanism}')
            alerts = sum(int(r['flagged']) for r in selected)
            tp = sum(int(r['true_positives']) for r in selected)
            trials = sum(int(r['trials']) for r in selected)
            output.append(dict(mechanism=mechanism, signal=signal, trials=trials,
                               alerts=alerts, true_positives=tp, precision=tp/alerts if alerts else None))
    target = [r for r in paired if r['transfer']=='within' and
              r['mechanism'].replace('-', '_')=='contextual_pairwise' and
              int(r['requests'])==500 and float(r['exploration'])==1.0]
    comparisons = {}
    for candidate in ('validation', 'combined'):
        selected = [r for r in target if r['candidate']==candidate]
        if len(selected)!=1:
            raise ValueError(f'expected one difficult-regime comparison for {candidate}')
        r = selected[0]
        comparisons[candidate] = {k:float(r[k]) for k in ('precision_difference','ci_low','ci_high')}
        comparisons[candidate]['seeds'] = int(r['seeds'])
    return output, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('reports/m72_extended'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m73_manuscript'))
    args = parser.parse_args(argv)
    summary_path = args.input_dir/'warning_summary.csv'
    paired_path = args.input_dir/'paired_precision.csv'
    summary = read_csv(summary_path, ('transfer','mode','mechanism','signal','flagged','true_positives','trials'))
    paired = read_csv(paired_path, ('transfer','mechanism','requests','exploration','candidate','precision_difference','ci_low','ci_high','seeds'))
    rows, comparisons = evidence(summary, paired)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'mechanism_precision.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    # Lazy import keeps the numerical audit usable without plotting dependencies.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    labels = ['ESS','Validation RMSE','Combined']
    xs = range(3)
    for offset, mechanism, label in [(-.18,'additive','Additive'),(.18,'contextual_pairwise','Contextual pairwise')]:
        values = [next(r['precision'] for r in rows if r['mechanism']==mechanism and r['signal']==signal)*100 for signal in ('ess','validation','combined')]
        ax.bar([x+offset for x in xs],values,width=.35,label=label)
    ax.set_xticks(list(xs),labels)
    ax.set_ylabel('Precision at matched 25% alert budget (%)')
    ax.set_title('Held-out warning performance depends on reward mechanism')
    ax.legend(); fig.tight_layout()
    fig.savefig(args.output_dir/'mechanism_reversal.png',dpi=180)
    plt.close(fig)
    manifest={'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path,paired_path)},
              'table':rows,'difficult_regime_paired':comparisons,
              'notes':['Pooled precision uses summed counts, not mean of cell precisions.',
                       'Matched-budget ranking uses held-out observable scores, not outcomes.',
                       'Seed-cluster bootstrap intervals are unadjusted for multiple comparisons.']}
    (args.output_dir/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Wrote mechanism_precision.csv, mechanism_reversal.png, evidence_manifest.json to {args.output_dir}')

if __name__=='__main__':
    main()
