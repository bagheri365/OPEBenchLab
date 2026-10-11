"""M6.5: seed-held-out calibration of observable OPE risk warnings."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m64_risk import Config, run

SIGNALS = ('ess', 'validation', 'combined')


def _score(row, signal, reference):
    """Calibration-only empirical percentile score (larger means riskier)."""
    def percentile(field, reverse=False):
        sample = np.asarray([r[field] for r in reference], dtype=float)
        value = float(row[field])
        # Midrank handles ESS ties, including many zero-match trials.
        if reverse:
            sample, value = -sample, -value
        return float((np.count_nonzero(sample < value) + .5*np.count_nonzero(sample == value))/len(sample))
    ess = percentile('test_ess', True)
    validation = percentile('validation_rmse')
    return {'ess': ess, 'validation': validation, 'combined': (ess+validation)/2}[signal]


def _threshold(scores, alert_fraction):
    # Strict comparison; threshold below minimum for an all-alert calibration.
    if alert_fraction == 1:
        return float(np.nextafter(min(scores), -np.inf))
    return float(np.quantile(scores, 1-alert_fraction))


def _metrics(rows):
    count = len(rows)
    flagged = sum(r['warning'] for r in rows)
    positives = sum(r['high_error'] for r in rows)
    tp = sum(r['warning'] and r['high_error'] for r in rows)
    fp = flagged-tp
    return dict(trials=count, flagged=flagged, high_error_count=positives,
                true_positives=tp, false_positives=fp,
                alert_rate=flagged/count if count else float('nan'),
                precision=tp/flagged if flagged else float('nan'),
                recall=tp/positives if positives else float('nan'),
                false_positive_rate=fp/(count-positives) if count>positives else float('nan'))


def _interval(rows, metric, bootstrap, rng):
    # Resample held-out SEED clusters, not individual trials; preserve pairing.
    seeds = sorted(set(r['seed'] for r in rows))
    if len(seeds) < 2 or bootstrap < 1:
        return (float('nan'), float('nan'))
    by_seed = {s: [r for r in rows if r['seed']==s] for s in seeds}
    estimates = []
    for _ in range(bootstrap):
        sample = [r for s in rng.choice(seeds, len(seeds), replace=True) for r in by_seed[int(s)]]
        value = _metrics(sample)[metric]
        if np.isfinite(value):
            estimates.append(value)
    return tuple(map(float, np.quantile(estimates, [.025,.975]))) if estimates else (float('nan'),float('nan'))


def evaluate(calibration, heldout, alert_fraction=.25, bootstrap=500, bootstrap_seed=2026):
    """Fit within-mechanism rules; also transfer contextual-pairwise rules to additive.

    Oracle error cutoffs are learned on calibration trials, never on held-out
    trials. Warning scores use only observable fields and calibration references.
    """
    if not 0 < alert_fraction <= 1:
        raise ValueError('alert_fraction must be in (0,1]')
    cal_seeds = {r['seed'] for r in calibration}
    test_seeds = {r['seed'] for r in heldout}
    if cal_seeds & test_seeds:
        raise ValueError('calibration and held-out seeds must be disjoint')
    groups = {}
    for row in calibration:
        groups.setdefault((row['mechanism'],row['requests'],row['exploration']), []).append(row)
    frozen, predictions, summary = [], [], []
    rng = np.random.default_rng(bootstrap_seed)
    for key, reference in sorted(groups.items()):
        mechanism, n, exploration = key
        error_cutoff = float(np.quantile([abs(r['dr_error']) for r in reference], .75))
        # For transfer, contextual-pairwise reference is applied to additive
        # outcomes at the same n/exploration; oracle cutoff stays frozen.
        destinations = [('within',mechanism)]
        if mechanism == 'contextual-pairwise':
            destinations.append(('pairwise-to-additive','additive'))
        for signal in SIGNALS:
            scores = [_score(r, signal, reference) for r in reference]
            cutoff = _threshold(scores, alert_fraction)
            frozen.append(dict(source_mechanism=mechanism, requests=n,
                               exploration=exploration, signal=signal,
                               score_threshold=cutoff, error_threshold=error_cutoff,
                               calibration_trials=len(reference), alert_fraction=alert_fraction))
            for transfer, destination in destinations:
                target = [r for r in heldout if (r['mechanism'],r['requests'],r['exploration']) == (destination,n,exploration)]
                if not target:
                    continue
                subset = []
                for row in target:
                    score = _score(row, signal, reference)
                    record = dict(source_mechanism=mechanism, mechanism=destination,
                                  transfer=transfer, requests=n, exploration=exploration,
                                  seed=row['seed'], trial=row['trial'], signal=signal,
                                  risk_score=score, score_threshold=cutoff,
                                  warning=int(score>cutoff),
                                  high_error=int(abs(row['dr_error'])>=error_cutoff),
                                  absolute_dr_error=abs(row['dr_error']),
                                  error_threshold=error_cutoff,
                                  test_ess=row['test_ess'], validation_rmse=row['validation_rmse'])
                    subset.append(record)
                predictions.extend(subset)
                result = dict(source_mechanism=mechanism, mechanism=destination,
                              transfer=transfer, requests=n, exploration=exploration,
                              signal=signal, **_metrics(subset))
                for metric in ('precision','recall','false_positive_rate','alert_rate'):
                    low, high = _interval(subset, metric, bootstrap, rng)
                    result[metric+'_ci_low'] = low
                    result[metric+'_ci_high'] = high
                summary.append(result)
    return frozen, predictions, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--requests', type=int, nargs='+', default=[500,2000,10000])
    p.add_argument('--explorations', type=float, nargs='+', default=[.5,1.])
    p.add_argument('--calibration-seeds', type=int, nargs='+', default=[42,1729,31415])
    p.add_argument('--heldout-seeds', type=int, nargs='+', default=[2718,1618,12345])
    p.add_argument('--ridges', type=float, nargs='+', default=[.01,1.,10.])
    p.add_argument('--alert-fraction', type=float, default=.25)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--bootstrap-seed', type=int, default=2026)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m65_calibration'))
    args = p.parse_args(argv)
    cal_seeds, test_seeds = tuple(args.calibration_seeds), tuple(args.heldout_seeds)
    if not cal_seeds or not test_seeds or set(cal_seeds)&set(test_seeds):
        p.error('calibration and held-out seeds must be nonempty and disjoint')
    if args.bootstrap < 0:
        p.error('--bootstrap must be nonnegative')
    shared = dict(trials=args.trials, requests=tuple(args.requests),
                  explorations=tuple(args.explorations), ridges=tuple(args.ridges))
    calibration = run(Config(seeds=cal_seeds, **shared))
    heldout = run(Config(seeds=test_seeds, **shared))
    frozen, predictions, summary = evaluate(calibration, heldout,
        args.alert_fraction, args.bootstrap, args.bootstrap_seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output/'calibration.json').write_text(json.dumps(dict(
        configuration=shared, calibration_seeds=cal_seeds, heldout_seeds=test_seeds,
        bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
        alert_fraction=args.alert_fraction, frozen_rules=frozen,
        score_definition='Calibration empirical midrank percentiles; ESS reversed; combined is mean of ESS and validation RMSE percentiles.',
        oracle_label='Absolute held-out DR error exceeds the frozen 75th percentile of calibration absolute DR errors for source mechanism.',
        selection_policy='No held-out oracle outcome influences score thresholds, score references, or error cutoffs.',
        caveats='Empirical quantile rules may not achieve exact held-out alert rates; score ties use strict >. Bootstrap resamples held-out seed clusters; few seeds give unstable intervals.'
    ), indent=2, default=list)+'\n')
    write_csv(output/'heldout_predictions.csv', predictions)
    write_csv(output/'heldout_summary.csv', summary)
    print(f'Wrote {len(predictions)} held-out decisions and {len(summary)} summaries to {output}')

if __name__ == '__main__':
    main()
