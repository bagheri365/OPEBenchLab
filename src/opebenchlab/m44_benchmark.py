"""M4.4: select ridge using logged-action out-of-fold Brier, never oracle truth.

Oracle-selected ridge is a simulation-only retrospective reference, not a
practical model-selection procedure. Selection and DR use the same logs, so
post-selection optimism remains possible; this is a research diagnostic.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opebenchlab.estimators import doubly_robust
from opebenchlab.m4_benchmark import write_csv
from opebenchlab.m42_benchmark import cross_fit, projection_matrix
from opebenchlab.reward_models import brier_score
from opebenchlab.synthetic import (SyntheticConfig, SyntheticWorld,
    epsilon_greedy_probabilities, generate_requests, log_interactions,
    oracle_policy_value, uniform_probabilities)


@dataclass(frozen=True)
class M44Config:
    trials: int = 30
    requests: tuple[int, ...] = (100, 500)
    explorations: tuple[float, ...] = (0.05, 0.2)
    ridges: tuple[float, ...] = (0.1, 10.0, 100.0)
    fixed_ridge: float = 10.0
    capacity: int = 40
    folds: int = 5
    seed: int = 42

    def __post_init__(self):
        if self.trials < 2 or self.folds < 2 or not self.requests or min(self.requests) < self.folds:
            raise ValueError('trials >= 2 and requests >= folds >= 2 required')
        if not self.explorations or any(not np.isfinite(x) or not 0 < x <= 1 for x in self.explorations):
            raise ValueError('explorations must be finite and in (0,1]')
        if not self.ridges or len(set(self.ridges)) != len(self.ridges) or any(not np.isfinite(x) or x <= 0 for x in self.ridges):
            raise ValueError('ridges must be unique positive finite values')
        if self.fixed_ridge not in self.ridges:
            raise ValueError('fixed_ridge must be among ridges')
        if self.capacity < 0:
            raise ValueError('capacity must be nonnegative')


def choose_by_brier(candidates):
    """Only consumes (ridge, observed-action OOF Brier) pairs."""
    return min(candidates, key=lambda x: (x[1], x[0]))[0]


def run(config: M44Config, progress=None):
    rows, choices = [], []
    projection = projection_matrix(config.capacity, config.seed + 777)
    total = config.trials * len(config.requests) * len(config.explorations)
    done = 0
    for n in config.requests:
        for exploration in config.explorations:
            for trial in range(config.trials):
                seed = config.seed + trial * 1009
                sim = SyntheticConfig(n_requests=n, exploration=exploration, seed=seed)
                world = SyntheticWorld(sim)
                requests = generate_requests(sim)
                logging = lambda r: epsilon_greedy_probabilities(r, exploration)
                data = log_interactions(world, requests, logging, seed=seed + 2)
                # Truth is obtained only after fitting and observable selection.
                candidates = {}
                for ridge in config.ridges:
                    model = cross_fit(data, projection, ridge, config.folds, seed + 3)
                    brier = brier_score(data, model)
                    estimate = doubly_robust(data, uniform_probabilities, model)
                    candidates[ridge] = (brier, estimate.value, estimate.effective_sample_size)
                selected = choose_by_brier([(r, candidates[r][0]) for r in config.ridges])
                truth = oracle_policy_value(world, requests, uniform_probabilities,
                                            f'm44-{n}-{trial}', 'uniform').true_expected_reward
                oracle = min(config.ridges, key=lambda r: ((candidates[r][1]-truth)**2, r))
                choices.append(dict(requests=n, exploration=exploration, trial=trial,
                                    fixed_ridge=config.fixed_ridge, brier_ridge=selected,
                                    oracle_ridge=oracle))
                for strategy, ridge in [('fixed', config.fixed_ridge),
                                        ('oof-brier', selected), ('oracle-reference', oracle)]:
                    brier, estimate, ess = candidates[ridge]
                    rows.append(dict(requests=n, exploration=exploration, trial=trial,
                                     strategy=strategy, selected_ridge=ridge, truth=truth,
                                     estimate=estimate, error=estimate-truth,
                                     squared_error=(estimate-truth)**2,
                                     brier=brier, ess=ess))
                done += 1
                if progress:
                    progress(done, total)
    return rows, choices


def summarize(rows):
    output = []
    keys = dict.fromkeys((r['requests'], r['exploration'], r['strategy']) for r in rows)
    for n, exploration, strategy in keys:
        group = [r for r in rows if (r['requests'], r['exploration'], r['strategy']) ==
                 (n, exploration, strategy)]
        errors = np.array([r['error'] for r in group])
        output.append(dict(requests=n, exploration=exploration, strategy=strategy,
                           trials=len(group), bias=float(errors.mean()),
                           rmse=float(np.sqrt(np.mean(errors**2))),
                           brier=float(np.mean([r['brier'] for r in group])),
                           ess=float(np.mean([r['ess'] for r in group]))))
    return output


def paired(rows):
    output = []
    keys = dict.fromkeys((r['requests'], r['exploration']) for r in rows)
    for n, exploration in keys:
        group = [r for r in rows if (r['requests'], r['exploration']) == (n, exploration)]
        by_trial = {}
        for r in group:
            by_trial.setdefault(r['trial'], {})[r['strategy']] = r['squared_error']
        for strategy in ('oof-brier', 'oracle-reference'):
            delta = np.array([v[strategy]-v['fixed'] for v in by_trial.values()])
            se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
            mean = float(delta.mean())
            output.append(dict(requests=n, exploration=exploration, strategy=strategy,
                               comparator='fixed', trials=len(delta), mean_delta_mse=mean,
                               ci_low=mean-1.96*se, ci_high=mean+1.96*se,
                               wins=int(np.sum(delta < 0))))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', type=int, nargs='+', default=[100, 500])
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.05, 0.2])
    parser.add_argument('--ridges', type=float, nargs='+', default=[0.1, 10, 100])
    parser.add_argument('--fixed-ridge', type=float, default=10)
    parser.add_argument('--capacity', type=int, default=40)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m44_selection'))
    args = parser.parse_args(argv)
    config = M44Config(args.trials, tuple(args.requests), tuple(args.explorations),
                       tuple(args.ridges), args.fixed_ridge, args.capacity, args.folds, args.seed)
    rows, choices = run(config, lambda done, total: print(f'Completed {done}/{total}', flush=True))
    summary = summarize(rows)
    for name, values in [('trials.csv', rows), ('summary.csv', summary),
                         ('paired.csv', paired(rows)), ('choices.csv', choices)]:
        write_csv(args.output_dir / name, values)
    for row in summary:
        print(f"n={row['requests']} eps={row['exploration']:.2f} "
              f"{row['strategy']:16s} RMSE={row['rmse']:.5f} Brier={row['brier']:.5f}")
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
