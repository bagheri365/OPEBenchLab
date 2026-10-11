"""M6.2 preregistered-style robustness suites for the M5.12 evaluation protocol.

This module deliberately reuses existing estimators; it never selects a method
using evaluation rewards. Suites vary one dimension at a time to control cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from opebenchlab.m511_validated import M511Config
from opebenchlab.m512_support import M512Config, run, support_regime
from opebenchlab.m52_benchmark import write_csv


@dataclass(frozen=True)
class Protocol:
    suite: str = 'all'
    trials: int = 10
    seeds: tuple[int, ...] = (42, 1729)
    requests: tuple[int, ...] = (500, 2000)
    explorations: tuple[float, ...] = (0.2, 1.0)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    candidates: int = 5
    k: int = 3
    noise: float = 0.05
    pair_bonus: float = 0.2
    low_ess_threshold: float = 10.0
    ci_level: float = 0.95

    def __post_init__(self):
        if self.suite not in ('all', 'replication', 'support', 'mechanism'):
            raise ValueError('unknown suite')
        if self.trials < 1 or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('invalid trials or seeds')
        if not 2 <= self.k <= self.candidates <= 7:
            raise ValueError('2 <= k <= candidates <= 7 required for exhaustive enumeration')
        if not 0 < self.ci_level < 1:
            raise ValueError('ci_level must be in (0,1)')
        # Validate the remainder using the source benchmark's constraints.
        M511Config(trials=self.trials, requests=self.requests, explorations=self.explorations,
                   ridges=self.ridges, candidates=self.candidates, k=self.k,
                   noise=self.noise, pair_bonus=self.pair_bonus)


def configurations(p: Protocol):
    """Fixed, nonadaptive grid; candidate count capped by source implementation."""
    suites = ('replication', 'support', 'mechanism') if p.suite == 'all' else (p.suite,)
    for suite in suites:
        if suite == 'replication':
            settings = [('contextual-pairwise', p.candidates, p.k, p.pair_bonus)]
        elif suite == 'support':
            settings = [('contextual-pairwise', c, k, p.pair_bonus)
                        for c, k in ((5, 2), (5, 3), (5, 4), (7, 3)) if k <= c]
        else:
            # Additive truth is a genuine zero-interaction negative control.
            # Contextual pairwise is the original synthetic mechanism.
            settings = [('additive', p.candidates, p.k, 0.0),
                        ('contextual-pairwise', p.candidates, p.k, p.pair_bonus)]
        for mechanism, candidates, k, bonus in settings:
            for seed in p.seeds:
                yield suite, mechanism, candidates, k, bonus, seed


def _finite_summary(rows):
    grouped = {}
    for row in rows:
        key = (row['suite'], row['mechanism'], row['candidates'], row['k'],
               row['requests'], row['exploration'], row['estimator'])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        errors = np.array([r['error'] for r in group], dtype=float)
        valid = errors[np.isfinite(errors)]
        result.append(dict(zip(('suite', 'mechanism', 'candidates', 'k', 'requests',
                                'exploration', 'estimator'), key),
                           trials=len(group), finite_trials=len(valid),
                           undefined_trials=len(group)-len(valid),
                           coverage=len(valid)/len(group),
                           conditional_rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else float('nan'),
                           conditional_bias=float(np.mean(valid)) if len(valid) else float('nan'),
                           zero_match_trials=sum(r['test_zero_match'] for r in group),
                           low_ess_trials=sum(r['test_support_regime'] == 'low-ess' for r in group),
                           fallback_trials=sum(r['fallback_used'] for r in group),
                           mean_test_ess=float(np.mean([r['test_ess'] for r in group]))))
    return result


def _intervals(rows, level):
    """Wald normal intervals for individual held-out OPE estimates.

    These are not bootstrap intervals for Monte Carlo estimator comparisons.
    SNIPS uses its ratio influence standard error; DR uses contribution SE.
    The source trial rows do not preserve held-out influence SEs, so interval
    coverage is explicitly marked unavailable rather than fabricated.
    """
    grouped = {}
    for row in rows:
        if row['estimator'] not in ('snips', 'contextual-dr'):
            continue
        key = (row['suite'], row['mechanism'], row['candidates'], row['k'],
               row['requests'], row['exploration'], row['estimator'])
        grouped.setdefault(key, []).append(row)
    return [dict(zip(('suite', 'mechanism', 'candidates', 'k', 'requests',
                      'exploration', 'estimator'), key), trials=len(group),
                 interval_method='unavailable: M5.11 does not export evaluation influence SE',
                 intervals_computed=0, covered=0, empirical_coverage=float('nan'),
                 mean_width=float('nan')) for key, group in sorted(grouped.items())]


def execute(p: Protocol):
    trials = []
    for suite, mechanism, candidates, k, bonus, seed in configurations(p):
        base = M511Config(trials=p.trials, requests=p.requests, explorations=p.explorations,
                          ridges=p.ridges, candidates=candidates, k=k, pair_bonus=bonus,
                          noise=p.noise, seed=seed)
        for row in run(M512Config(base, p.low_ess_threshold)):
            row = dict(row)
            row.update(suite=suite, mechanism=mechanism, candidates=candidates,
                       k=k, replicate_seed=seed)
            # Trial keys are composite: (suite, mechanism, dimensions, seed, trial).
            trials.append(row)
    return trials, _finite_summary(trials), _intervals(trials, p.ci_level)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', 'replication', 'support', 'mechanism'), default='all')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1729])
    parser.add_argument('--requests', nargs='+', type=int, default=[500, 2000])
    parser.add_argument('--explorations', nargs='+', type=float, default=[0.2, 1.0])
    parser.add_argument('--ridges', nargs='+', type=float, default=[0.01, 1, 10])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--pair-bonus', type=float, default=0.2)
    parser.add_argument('--low-ess-threshold', type=float, default=10)
    parser.add_argument('--ci-level', type=float, default=0.95)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m62_robustness'))
    a = parser.parse_args(argv)
    p = Protocol(a.suite, a.trials, tuple(a.seeds), tuple(a.requests),
                 tuple(a.explorations), tuple(a.ridges), a.candidates, a.k,
                 a.noise, a.pair_bonus, a.low_ess_threshold, a.ci_level)
    trials, summary, intervals = execute(p)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('robustness_trials.csv', trials),
                       ('robustness_summary.csv', summary),
                       ('interval_coverage.csv', intervals)):
        write_csv(a.output_dir / name, rows)
    protocol = dict(asdict(p), estimator_implementations='unchanged M5.11/M5.12',
                    mechanism_scope='additive vs original contextual-pairwise only; no misspecified-interaction mechanism',
                    interval_limitation='evaluation influence SE unavailable in M5.11 output',
                    multiplicity='exploratory; no multiple-testing adjustment',
                    selection='SNIPS-first, fallback only at zero evaluation ESS; uses actions, not rewards',
                    analysis='coverage and conditional RMSE; nonfinite trials retained as failures')
    protocol['protocol_sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    (a.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(f'Wrote {len(trials)} trial-estimator rows, {len(summary)} summary rows to {a.output_dir}')


if __name__ == '__main__':
    main()
