"""M5.12: support-aware slate OPE with explicit estimator coverage.

The fallback reads final-evaluation action/propensity support (ESS), not rewards.
The M5.11 validation-only selector remains separately reported.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from opebenchlab.m511_validated import M511Config, run as run_m511
from opebenchlab.m52_benchmark import write_csv


@dataclass(frozen=True)
class M512Config:
    base: M511Config = M511Config()
    low_ess_threshold: float = 10.0

    def __post_init__(self):
        if not np.isfinite(self.low_ess_threshold) or self.low_ess_threshold < 0:
            raise ValueError('low_ess_threshold must be finite and nonnegative')


def support_regime(ess: float, threshold: float) -> str:
    if ess <= 0:
        return 'zero-match'
    return 'low-ess' if ess < threshold else 'adequate'


def run(config: M512Config) -> list[dict]:
    """Reuse paired M5.11 trials; choose fallback using only evaluation ESS.

    The M5.11 estimator itself is selected on validation data.  This additional
    fallback explicitly inspects evaluation *actions*, never evaluation rewards.
    """
    source = run_m511(config.base)
    grouped = {}
    for row in source:
        key = (row['requests'], row['exploration'], row['trial'])
        grouped.setdefault(key, {})[row['estimator']] = row
    output = []
    for group in grouped.values():
        ref = group['snips']
        ess = float(ref['test_ess'])
        regime = support_regime(ess, config.low_ess_threshold)
        chosen = 'contextual-dr' if ess == 0 else 'snips'
        for estimator in ('snips', 'contextual-dr', 'contextual-dm', 'selected', 'support-aware'):
            source_row = group[chosen] if estimator == 'support-aware' else group[estimator]
            value = float(source_row['estimate'])
            row = dict(source_row)
            row.update(estimator=estimator, estimate=value,
                       error=value-float(ref['true_value']),
                       finite=bool(np.isfinite(value)),
                       test_support_regime=regime,
                       test_zero_match=bool(ess == 0),
                       fallback_used=bool(estimator == 'support-aware' and ess == 0),
                       support_selection_source=('evaluation-actions-only' if estimator == 'support-aware'
                                                 else 'validation-only' if estimator == 'selected'
                                                 else 'fixed-estimator'))
            # M5.11 oracle regret can be NaN when SNIPS is undefined; don't inherit it.
            row.pop('squared_regret', None)
            output.append(row)
    return output


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row['requests'], row['exploration'], row['estimator'])
        grouped.setdefault(key, []).append(row)
    out = []
    for (n, eps, estimator), group in sorted(grouped.items()):
        errors = np.array([r['error'] for r in group], dtype=float)
        finite = np.isfinite(errors)
        valid = errors[finite]
        # Paired SNIPS-vs-DR differences are computed on the intersection of
        # finite estimates, never by independently dropping different trials.
        out.append(dict(requests=n, exploration=eps, estimator=estimator,
                        trials=len(group), finite_trials=int(finite.sum()),
                        undefined_trials=int((~finite).sum()),
                        coverage=float(finite.mean()),
                        conditional_bias=float(np.mean(valid)) if len(valid) else float('nan'),
                        conditional_rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else float('nan'),
                        zero_match_trials=sum(r['test_zero_match'] for r in group),
                        low_ess_trials=sum(r['test_support_regime'] == 'low-ess' for r in group),
                        fallback_trials=sum(r['fallback_used'] for r in group)))
    return out


def paired_comparison(rows: list[dict]) -> list[dict]:
    grouped = {}
    for r in rows:
        grouped.setdefault((r['requests'], r['exploration'], r['trial']), {})[r['estimator']] = r
    bins = {}
    for (n, eps, _), g in grouped.items():
        if not {'snips', 'contextual-dr'} <= g.keys():
            continue
        a, b = g['snips']['error'], g['contextual-dr']['error']
        bins.setdefault((n, eps), []).append((a, b))
    result = []
    for (n, eps), pairs in sorted(bins.items()):
        valid = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        differences = np.array([b*b-a*a for a, b in valid])
        result.append(dict(requests=n, exploration=eps, paired_trials=len(valid),
                           excluded_nonfinite=len(pairs)-len(valid),
                           mean_dr_minus_snips_squared_error=float(differences.mean()) if len(valid) else float('nan'),
                           dr_win_fraction=float(np.mean(differences < 0)) if len(valid) else float('nan')))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    p.add_argument('--explorations', type=float, nargs='+', default=[0.2, 1.0])
    p.add_argument('--ridges', type=float, nargs='+', default=[0.01, 1, 10])
    p.add_argument('--candidates', type=int, default=5)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--pair-bonus', type=float, default=0.2)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--train-fraction', type=float, default=0.4)
    p.add_argument('--validation-fraction', type=float, default=0.25)
    p.add_argument('--variance-ratio', type=float, default=0.8)
    p.add_argument('--agreement-z', type=float, default=1.0)
    p.add_argument('--low-ess-threshold', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m512_support'))
    a = p.parse_args(argv)
    base = M511Config(trials=a.trials, requests=tuple(a.requests), explorations=tuple(a.explorations),
                      ridges=tuple(a.ridges), candidates=a.candidates, k=a.k,
                      pair_bonus=a.pair_bonus, noise=a.noise,
                      validation_fraction=a.validation_fraction, train_fraction=a.train_fraction,
                      variance_ratio=a.variance_ratio, agreement_z=a.agreement_z, seed=a.seed)
    rows = run(M512Config(base, a.low_ess_threshold))
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summarize(rows))
    write_csv(a.output_dir / 'paired.csv', paired_comparison(rows))
    print(f'Wrote {len(rows)} rows to {a.output_dir}')


if __name__ == '__main__':
    main()
