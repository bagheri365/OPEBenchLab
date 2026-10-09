"""M5.2: Monte Carlo overlap and error benchmark for exact full-slate IPS/SNIPS."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path

import numpy as np

from opebenchlab.m51_slates import log_slates, slate_ips, slate_snips, slate_weights
from opebenchlab.synthetic import SyntheticConfig


@dataclass(frozen=True)
class M52Config:
    trials: int = 30
    requests: tuple[int, ...] = (500, 2000, 10000)
    ks: tuple[int, ...] = (1, 2, 3, 4)
    explorations: tuple[float, ...] = (0.05, 0.2, 0.5, 1.0)
    candidates: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.trials < 1 or self.candidates < 1 or not self.requests or not self.ks or not self.explorations:
            raise ValueError('positive trials/candidates and nonempty parameter grids required')
        if any(n < 1 for n in self.requests) or any(k < 1 or k > self.candidates for k in self.ks):
            raise ValueError('invalid requests or K')
        if any(not isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('explorations must be finite and in (0, 1]')


def run(config: M52Config, progress=None) -> list[dict]:
    """One independently seeded synthetic world/log per configuration and trial."""
    rows = []
    total = len(config.requests) * len(config.ks) * len(config.explorations) * config.trials
    for n in config.requests:
        for k in config.ks:
            for exploration in config.explorations:
                for trial in range(config.trials):
                    # Independent seeds across configurations, reproducible without Python hash randomization.
                    seed = config.seed + trial + 100003 * n + 1009 * k + int(round(exploration * 1000000)) * 10000019
                    data, oracle = log_slates(SyntheticConfig(n_requests=n, candidates_per_request=config.candidates, exploration=exploration, seed=seed), k)
                    ips = slate_ips(data, k)
                    weights = slate_weights(data, k)
                    matches = int(np.count_nonzero(weights))
                    try:
                        snips_value = slate_snips(data, k).value
                    except ValueError as exc:
                        if 'no exact target slate observed' not in str(exc):
                            raise
                        snips_value = float('nan')
                    for name, value in (('IPS', ips.value), ('SNIPS', snips_value)):
                        defined = bool(np.isfinite(value))
                        rows.append(dict(requests=n, k=k, exploration=exploration, trial=trial, estimator=name,
                                         oracle=oracle, estimate=value, error=value - oracle if defined else float('nan'),
                                         squared_error=(value - oracle) ** 2 if defined else float('nan'),
                                         ess=ips.effective_sample_size, max_weight=ips.max_weight,
                                         matches=matches, match_rate=matches / n, zero_match=int(matches == 0),
                                         undefined=int(not defined)))
                    if progress:
                        progress(len(rows) // 2, total)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    groups = {}
    for row in rows:
        key = (row['requests'], row['k'], row['exploration'], row['estimator'])
        groups.setdefault(key, []).append(row)
    result = []
    for (n, k, exploration, estimator), group in sorted(groups.items()):
        valid = [r for r in group if not r['undefined']]
        # Undefined SNIPS is excluded from conditional error metrics, never silently set to zero.
        result.append(dict(requests=n, k=k, exploration=exploration, estimator=estimator,
                           trials=len(group), defined_trials=len(valid),
                           bias=float(np.mean([r['error'] for r in valid])) if valid else float('nan'),
                           rmse=sqrt(float(np.mean([r['squared_error'] for r in valid]))) if valid else float('nan'),
                           mean_ess=float(np.mean([r['ess'] for r in group])),
                           mean_max_weight=float(np.mean([r['max_weight'] for r in group])),
                           mean_matches=float(np.mean([r['matches'] for r in group])),
                           mean_match_rate=float(np.mean([r['match_rate'] for r in group])),
                           zero_match_frequency=float(np.mean([r['zero_match'] for r in group])),
                           undefined_frequency=float(np.mean([r['undefined'] for r in group]))))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', nargs='+', type=int, default=[500, 2000, 10000])
    parser.add_argument('--ks', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--explorations', nargs='+', type=float, default=[0.05, 0.2, 0.5, 1.0])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m52_slates'))
    args = parser.parse_args(argv)
    config = M52Config(args.trials, tuple(args.requests), tuple(args.ks), tuple(args.explorations), args.candidates, args.seed)
    rows = run(config, progress=lambda done, total: print(f'Completed {done}/{total}', flush=True) if done % 10 == 0 or done == total else None)
    summary = summarize(rows)
    write_csv(args.output_dir / 'trials.csv', rows)
    write_csv(args.output_dir / 'summary.csv', summary)
    print('requests  K  exploration  estimator  RMSE      ESS    match-rate  zero-match  undefined')
    for r in summary:
        print(f"{r['requests']:8d} {r['k']:2d} {r['exploration']:12.2f} {r['estimator']:>10s} {r['rmse']:8.4f} {r['mean_ess']:8.1f} {r['mean_match_rate']:11.4f} {r['zero_match_frequency']:11.3f} {r['undefined_frequency']:10.3f}")
    print(f'Wrote {args.output_dir / "summary.csv"} and {args.output_dir / "trials.csv"}')


if __name__ == '__main__':
    main()
