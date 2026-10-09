"""M4.3: ridge regularization, sample-size learning curves, and honest evaluation.

All estimators in a trial use the same evaluation logs. The external model
trains on independent requests and rewards; the oracle is scoring-only.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opebenchlab.estimators import doubly_robust
from opebenchlab.m4_benchmark import write_csv
from opebenchlab.m42_benchmark import (fit, cross_fit, projection_matrix,
                                      summarize, paired_differences)
from opebenchlab.reward_models import brier_score
from opebenchlab.synthetic import (SyntheticConfig, SyntheticWorld,
    epsilon_greedy_probabilities, generate_requests, log_interactions,
    oracle_policy_value, uniform_probabilities)


@dataclass(frozen=True)
class M43Config:
    trials: int = 30
    requests: tuple[int, ...] = (100, 500)
    explorations: tuple[float, ...] = (0.05, 0.2)
    ridges: tuple[float, ...] = (0.1, 10.0, 100.0)
    capacity: int = 40
    folds: int = 5
    seed: int = 42

    def __post_init__(self):
        if self.trials < 2 or self.folds < 2 or not self.requests or min(self.requests) < self.folds:
            raise ValueError('trials >= 2 and requests >= folds >= 2 required')
        if not self.explorations or any(not np.isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('explorations must be in (0, 1]')
        if not self.ridges or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('ridges must be positive and finite')
        if self.capacity < 0:
            raise ValueError('capacity must be nonnegative')


def run(config: M43Config, progress=None):
    rows = []
    total = config.trials * len(config.requests) * len(config.explorations)
    done = 0
    projection = projection_matrix(config.capacity, config.seed + 777)
    for n in config.requests:
        for exploration in config.explorations:
            for trial in range(config.trials):
                seed = config.seed + trial * 1009
                sim = SyntheticConfig(n_requests=n, exploration=exploration, seed=seed)
                world = SyntheticWorld(sim)
                requests = generate_requests(sim)
                logging = lambda r: epsilon_greedy_probabilities(r, exploration)
                data = log_interactions(world, requests, logging, seed=seed + 2)
                truth = oracle_policy_value(world, requests, uniform_probabilities,
                                            f'm43-{n}-{trial}', 'uniform').true_expected_reward
                # Independent request set and reward draws, same data-generating world.
                train_sim = SyntheticConfig(n_requests=n, exploration=exploration, seed=seed + 100000)
                external_requests = generate_requests(train_sim)
                external = log_interactions(world, external_requests, logging, seed=seed + 100002)
                for ridge in config.ridges:
                    models = (('in-sample', fit(data, projection, ridge)),
                              ('cross-fitted', cross_fit(data, projection, ridge, config.folds, seed + 3)),
                              ('external', fit(external, projection, ridge)))
                    for strategy, model in models:
                        estimate = doubly_robust(data, uniform_probabilities, model)
                        rows.append(dict(requests=n, exploration=exploration, trial=trial,
                                         capacity=config.capacity, ridge=ridge, strategy=strategy,
                                         truth=truth, estimate=estimate.value,
                                         error=estimate.value-truth,
                                         squared_error=(estimate.value-truth)**2,
                                         brier=brier_score(data, model),
                                         ess=estimate.effective_sample_size))
                done += 1
                if progress:
                    progress(done, total)
    return rows


def summarize_m43(rows):
    result = []
    for ridge in dict.fromkeys(r['ridge'] for r in rows):
        selected = [r for r in rows if r['ridge'] == ridge]
        for item in summarize(selected):
            result.append(dict(ridge=ridge, **item))
    return result


def paired_m43(rows):
    result = []
    for ridge in dict.fromkeys(r['ridge'] for r in rows):
        selected = [r for r in rows if r['ridge'] == ridge and r['strategy'] != 'external']
        for item in paired_differences(selected):
            result.append(dict(ridge=ridge, **item))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', type=int, nargs='+', default=[100, 500])
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.05, 0.2])
    parser.add_argument('--ridges', type=float, nargs='+', default=[0.1, 10, 100])
    parser.add_argument('--capacity', type=int, default=40)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m43_regularization'))
    args = parser.parse_args(argv)
    config = M43Config(args.trials, tuple(args.requests), tuple(args.explorations),
                       tuple(args.ridges), args.capacity, args.folds, args.seed)
    rows = run(config, lambda done, total: print(f'Completed {done}/{total}', flush=True))
    summary, paired = summarize_m43(rows), paired_m43(rows)
    for filename, values in [('trials.csv', rows), ('summary.csv', summary), ('paired.csv', paired)]:
        write_csv(args.output_dir / filename, values)
    for row in summary:
        print(f"n={row['requests']} eps={row['exploration']:.2f} ridge={row['ridge']:g} "
              f"{row['strategy']:13s} RMSE={row['rmse']:.5f} Brier={row['brier']:.5f}")
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
