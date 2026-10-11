"""M7.1: reproducible publication figures from archived CSV outputs.

No simulator oracle values enter plotting beyond already reported experiment metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

INPUTS = {
    'paired': 'm61_support/paired_bootstrap.csv',
    'coverage': 'm61_support/coverage.csv',
    'replication': 'm62_replication/robustness_summary.csv',
    'support': 'm62_support/robustness_summary.csv',
    'mechanism': 'm62_mechanism/robustness_summary.csv',
    'warnings': 'm65_calibration/heldout_summary.csv',
}
REQUIRED = {
    'paired': {'requests', 'exploration', 'paired_trials', 'mean_dr_minus_snips_squared_error', 'bootstrap_ci_lower', 'bootstrap_ci_upper'},
    'coverage': {'estimator', 'trials', 'finite_trials', 'undefined_trials'},
    'replication': {'requests', 'exploration', 'estimator', 'conditional_rmse', 'coverage', 'trials'},
    'support': {'requests', 'exploration', 'candidates', 'k', 'estimator', 'conditional_rmse', 'coverage', 'mean_test_ess', 'trials'},
    'mechanism': {'mechanism', 'requests', 'exploration', 'estimator', 'conditional_rmse', 'trials'},
    'warnings': {'source_mechanism', 'mechanism', 'transfer', 'signal', 'trials', 'flagged', 'high_error_count', 'true_positives'},
}


def load_reports(root: Path):
    tables, provenance = {}, {}
    for key, relative in INPUTS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Missing {key} input: {path}')
        raw = path.read_bytes()
        with path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED[key] - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'{path}: missing columns: {sorted(missing)}')
            tables[key] = list(reader)
        if not tables[key]:
            raise ValueError(f'{path}: empty input')
        provenance[key] = {'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(tables[key])}
    return tables, provenance


def select(table, **conditions):
    return [row for row in table if all(str(row[k]) == str(v) for k, v in conditions.items())]


def one(table, **conditions):
    matches = select(table, **conditions)
    if len(matches) != 1:
        raise ValueError(f'Expected one row for {conditions}; got {len(matches)}')
    return matches[0]


def build_data(tables):
    """Build all plotted values from input rows, never hardcoded result numbers."""
    replication = []
    for n in (500, 2000, 10000):
        for estimator in ('snips', 'contextual-dr'):
            r = one(tables['replication'], requests=n, exploration=1.0, estimator=estimator)
            replication.append({'requests': n, 'estimator': estimator, 'rmse': float(r['conditional_rmse']), 'coverage': float(r['coverage']), 'trials': int(r['trials'])})
    support = []
    for candidates, k in ((5, 2), (5, 3), (5, 4), (7, 3)):
        for estimator in ('snips', 'support-aware'):
            r = one(tables['support'], candidates=candidates, k=k, requests=500, exploration=1.0, estimator=estimator)
            support.append({'candidates': candidates, 'k': k, 'estimator': estimator, 'coverage': float(r['coverage']), 'rmse': float(r['conditional_rmse']), 'ess': float(r['mean_test_ess']), 'trials': int(r['trials'])})
    mechanism = []
    for mech in ('additive', 'contextual-pairwise'):
        for n in (500, 2000):
            for estimator in ('snips', 'contextual-dr'):
                r = one(tables['mechanism'], mechanism=mech, requests=n, exploration=1.0, estimator=estimator)
                mechanism.append({'mechanism': mech, 'requests': n, 'estimator': estimator, 'rmse': float(r['conditional_rmse']), 'trials': int(r['trials'])})
    warnings = []
    for signal in ('ess', 'validation', 'combined'):
        rows = select(tables['warnings'], transfer='within', signal=signal)
        if len(rows) != 12:
            raise ValueError(f'Expected 12 within-mechanism rows for {signal}, got {len(rows)}')
        alerts = sum(int(r['flagged']) for r in rows)
        trials = sum(int(r['trials']) for r in rows)
        positives = sum(int(r['high_error_count']) for r in rows)
        tp = sum(int(r['true_positives']) for r in rows)
        warnings.append({'signal': signal, 'trials': trials, 'alerts': alerts, 'high_errors': positives, 'true_positives': tp, 'precision': tp / alerts if alerts else None, 'recall': tp / positives if positives else None, 'alert_rate': alerts / trials})
    coverage = {}
    for estimator in ('snips', 'support-aware'):
        rows = select(tables['coverage'], estimator=estimator)
        if not rows:
            raise ValueError(f'No coverage rows for {estimator}')
        total = sum(int(r['trials']) for r in rows)
        finite = sum(int(r['finite_trials']) for r in rows)
        undefined = sum(int(r['undefined_trials']) for r in rows)
        if total != finite + undefined:
            raise ValueError(f'Coverage counts inconsistent for {estimator}')
        coverage[estimator] = {'trials': total, 'finite': finite, 'undefined': undefined}
    paired = []
    for n in (500, 2000):
        r = one(tables['paired'], requests=n, exploration=1.0)
        paired.append({'requests': n, 'paired_trials': int(r['paired_trials']), 'mean_difference': float(r['mean_dr_minus_snips_squared_error']), 'ci_low': float(r['bootstrap_ci_lower']), 'ci_high': float(r['bootstrap_ci_upper'])})
    return {'replication': replication, 'support': support, 'mechanism': mechanism, 'warnings': warnings, 'coverage': coverage, 'paired': paired}


def make_figures(data, output: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Figures require matplotlib; install with pip install matplotlib') from exc
    paths = []
    def save(name):
        plt.tight_layout()
        path = output / name
        plt.savefig(path, dpi=180, bbox_inches='tight')
        plt.close()
        paths.append(name)

    fig, ax = plt.subplots(figsize=(7, 4))
    for estimator in ('snips', 'contextual-dr'):
        rows = [r for r in data['replication'] if r['estimator'] == estimator]
        ax.plot([r['requests'] for r in rows], [r['rmse'] for r in rows], marker='o', label=estimator)
    ax.set(xlabel='Logged requests', ylabel='RMSE (conditional on finite estimate)', title='M6.2 replication · exploration 1.0')
    ax.legend(); ax.grid(alpha=.25)
    save('m71_replication_rmse.png')

    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [r for r in data['support'] if r['estimator'] == 'snips']
    labels = [f"{r['candidates']} items, K={r['k']}" for r in rows]
    ax.plot(labels, [100*r['coverage'] for r in rows], marker='o', label='SNIPS coverage')
    ax.set(xlabel='Slate configuration', ylabel='Finite-estimate coverage (%)', ylim=(0, 105), title='M6.2 support stress · n=500, exploration 1.0')
    ax.tick_params(axis='x', rotation=15); ax.grid(alpha=.25)
    save('m71_support_coverage.png')

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [('additive', 500), ('contextual-pairwise', 500), ('additive', 2000), ('contextual-pairwise', 2000)]
    x = list(range(len(groups)))
    for offset, estimator in ((-.18, 'snips'), (.18, 'contextual-dr')):
        vals = [next(r['rmse'] for r in data['mechanism'] if r['mechanism'] == m and r['requests'] == n and r['estimator'] == estimator) for m, n in groups]
        ax.bar([i+offset for i in x], vals, width=.35, label=estimator)
    ax.set_xticks(x, [f'{m}\nn={n}' for m,n in groups]); ax.set(ylabel='RMSE (conditional on finite estimate)', title='M6.2 reward mechanism · exploration 1.0')
    ax.legend(); save('m71_mechanism_rmse.png')

    fig, ax = plt.subplots(figsize=(7, 4))
    warnings = data['warnings']
    ax.bar([r['signal'] for r in warnings], [100*r['precision'] for r in warnings])
    baseline = 100*warnings[0]['high_errors']/warnings[0]['trials']
    ax.axhline(baseline, linestyle='--', label=f'High-error prevalence ({baseline:.1f}%)')
    ax.set(ylabel='Held-out precision (%)', title='M6.5 within-mechanism warnings (unmatched alert rates)')
    ax.legend(); save('m71_heldout_precision.png')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-dir', type=Path, required=True, help='Directory containing m61_support, m62_* and m65_calibration')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    tables, provenance = load_reports(args.reports_dir)
    data = build_data(tables)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = make_figures(data, args.output_dir)
    manifest = {'protocol': 'M7.1 publication figures v1', 'sources': provenance, 'derived': data, 'figures': images, 'limitations': ['SNIPS RMSE conditions on finite estimates', 'M6.5 warning alert rates differ', 'Only three M6.5 held-out seeds', 'Bootstrap comparisons are not multiplicity-adjusted']}
    (args.output_dir / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'Generated {len(images)} figures and figure_manifest.json in {args.output_dir}')


if __name__ == '__main__':
    main()
