"""M7.2: paired, seed-cluster validation of frozen and matched-budget warnings.

Reads M6.5 heldout_predictions.csv; never fits risk scores using oracle labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SIGNALS = ('ess', 'validation', 'combined')
KEYS = ('source_mechanism', 'mechanism', 'transfer', 'requests', 'exploration')


def load(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = set(KEYS) | {'seed', 'trial', 'signal', 'risk_score', 'warning', 'high_error'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('missing or empty M6.5 heldout predictions; required: ' + ', '.join(sorted(required)))
    for row in rows:
        row['seed'] = int(row['seed'])
        row['trial'] = int(row['trial'])
        row['risk_score'] = float(row['risk_score'])
        row['warning'] = int(row['warning'])
        row['high_error'] = int(row['high_error'])
        if row['warning'] not in (0, 1) or row['high_error'] not in (0, 1):
            raise ValueError('warning and high_error must be binary')
        if not np.isfinite(row['risk_score']):
            raise ValueError('non-finite risk score')
    return rows


def _metrics(records, decisions):
    labels = [int(r['high_error']) for r in records]
    tp = sum(a and y for a, y in zip(decisions, labels))
    flagged = sum(decisions)
    positives = sum(labels)
    n = len(records)
    return {'trials': n, 'flagged': flagged, 'positives': positives,
            'true_positives': tp, 'alert_rate': flagged/n,
            'precision': tp/flagged if flagged else float('nan'),
            'recall': tp/positives if positives else float('nan'),
            'false_positive_rate': (flagged-tp)/(n-positives) if n>positives else float('nan')}


def _ranked_decisions(rows, budget):
    """Fixed alert count using only risk scores; stable ID tie-break independent of labels."""
    count = min(len(rows), max(0, int(np.floor(len(rows)*budget + .5))))
    ordering = sorted(range(len(rows)), key=lambda i: (-rows[i]['risk_score'], rows[i]['seed'], rows[i]['trial']))
    selected = set(ordering[:count])
    return [int(i in selected) for i in range(len(rows))]


def analyze(rows, budget=.25, bootstrap=1000, seed=2026):
    if not 0 < budget <= 1 or bootstrap < 0:
        raise ValueError('budget must be in (0,1] and bootstrap nonnegative')
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        ident = (row['seed'], row['trial'])
        if ident in groups[key].setdefault(row['signal'], {}):
            raise ValueError('duplicate trial/signal')
        groups[key][row['signal']][ident] = row
    rng = np.random.default_rng(seed)
    summary, paired, decisions = [], [], []
    for key, signals in sorted(groups.items()):
        if set(signals) != set(SIGNALS):
            raise ValueError('incomplete warning signals for group ' + str(key))
        ids = sorted(signals['ess'])
        if not ids or any(set(signals[s]) != set(ids) for s in SIGNALS):
            raise ValueError('unpaired trial IDs across signals')
        labels = [signals['ess'][i]['high_error'] for i in ids]
        if any(signals[s][i]['high_error'] != labels[j] for s in SIGNALS for j,i in enumerate(ids)):
            raise ValueError('inconsistent oracle labels across paired signals')
        by_signal = {s: [signals[s][i] for i in ids] for s in SIGNALS}
        ranked = {s: _ranked_decisions(by_signal[s], budget) for s in SIGNALS}
        frozen = {s: [r['warning'] for r in by_signal[s]] for s in SIGNALS}
        for mode, flags in [('frozen', frozen), ('matched', ranked)]:
            for s in SIGNALS:
                metrics = _metrics(by_signal[s], flags[s])
                summary.append(dict(zip(KEYS,key), mode=mode, signal=s, **metrics))
                for ident, row, flag in zip(ids, by_signal[s], flags[s]):
                    decisions.append(dict(zip(KEYS,key), seed=ident[0], trial=ident[1],
                                          mode=mode, signal=s, warning=flag,
                                          high_error=row['high_error'], risk_score=row['risk_score']))
            # Pairwise precision at matched budget; frozen thresholds have variable denominators.
            if mode != 'matched':
                continue
            base = _metrics(by_signal['ess'], flags['ess'])['precision']
            seeds = sorted({i[0] for i in ids})
            seed_positions = {s: [j for j, ident in enumerate(ids) if ident[0] == s] for s in seeds}
            for candidate in ('validation', 'combined'):
                observed = _metrics(by_signal[candidate], flags[candidate])['precision'] - base
                draws = []
                if len(seeds) >= 2:
                    for _ in range(bootstrap):
                        positions = [j for s in rng.choice(seeds, len(seeds), replace=True) for j in seed_positions[int(s)]]
                        # Preserve the fixed, original held-out decisions during cluster resampling.
                        def precision(signal):
                            selected = [j for j in positions if flags[signal][j]]
                            return sum(labels[j] for j in selected)/len(selected) if selected else float('nan')
                        difference = precision(candidate)-precision('ess')
                        if np.isfinite(difference):
                            draws.append(difference)
                lo, hi = (np.quantile(draws,[.025,.975]) if draws else (float('nan'),float('nan')))
                paired.append(dict(zip(KEYS,key), candidate=candidate, baseline='ess',
                                   matched_budget=budget, precision_difference=float(observed),
                                   ci_low=float(lo), ci_high=float(hi), seeds=len(seeds),
                                   bootstrap_draws=len(draws)))
    return summary, paired, decisions


def _write_csv(path, rows):
    if not rows:
        raise ValueError('no output rows')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('reports/m65_calibration'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m72_validation'))
    parser.add_argument('--alert-fraction', type=float, default=.25)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    rows = load(args.input_dir/'heldout_predictions.csv')
    summary, paired, decisions = analyze(rows, args.alert_fraction, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir/'warning_summary.csv', summary)
    _write_csv(args.output_dir/'paired_precision.csv', paired)
    _write_csv(args.output_dir/'warning_decisions.csv', decisions)
    (args.output_dir/'protocol.json').write_text(json.dumps(dict(
        input=str(args.input_dir/'heldout_predictions.csv'), alert_fraction=args.alert_fraction,
        bootstrap=args.bootstrap, bootstrap_seed=args.seed,
        matched_budget='Per mechanism/request/exploration/transfer group, choose round-half-up(n * budget) highest observable scores, with deterministic seed/trial tie break.',
        inference='Seed-cluster bootstrap of fixed held-out alert decisions; intervals describe precision differences of those decisions, not re-ranked policies.',
        caution='Matched-budget ranks use held-out observable score distribution (transductive); not deployable frozen thresholds. Bootstrap with few independent seeds is unstable. More seeds require rerunning M6.5.',
        oracle_isolation='High-error labels used only for scoring, never for selecting matched-budget alerts.'
    ), indent=2)+'\n')
    print(f'Wrote {len(summary)} summaries, {len(paired)} paired comparisons, {len(decisions)} decisions')


if __name__ == '__main__':
    main()
