"""M4.5: independent ridge selection with an untouched DR evaluation partition.

The OOF-on-evaluation strategy is diagnostic and does not satisfy independence.
Oracle truth is used exclusively for scoring after model selection and fitting.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from opebenchlab.synthetic import (SyntheticConfig, SyntheticWorld, LoggedDataset,
    epsilon_greedy_probabilities, generate_requests, log_interactions,
    oracle_policy_value, uniform_probabilities)
from opebenchlab.m42_benchmark import fit, cross_fit, projection_matrix
from opebenchlab.m44_benchmark import choose_by_brier, summarize, paired
from opebenchlab.reward_models import brier_score
from opebenchlab.estimators import doubly_robust
from opebenchlab.m4_benchmark import write_csv


@dataclass(frozen=True)
class M45Config:
    trials: int = 30
    requests: tuple[int, ...] = (100, 500)
    explorations: tuple[float, ...] = (0.05, 0.2)
    ridges: tuple[float, ...] = (0.1, 10.0, 100.0)
    fixed_ridge: float = 10.0
    capacity: int = 40
    folds: int = 5
    train_fraction: float = 0.5
    seed: int = 42

    def __post_init__(self):
        if self.trials < 2 or self.folds < 2 or not self.requests:
            raise ValueError('trials >= 2 and folds >= 2 required')
        if not 0 < self.train_fraction < 1 or not np.isfinite(self.train_fraction):
            raise ValueError('train_fraction must be in (0,1)')
        if any(min(int(n * self.train_fraction), n - int(n * self.train_fraction)) < self.folds for n in self.requests):
            raise ValueError('both partitions must have at least folds requests')
        if not self.explorations or any(not np.isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('invalid explorations')
        if not self.ridges or len(set(self.ridges)) != len(self.ridges) or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('invalid ridges')
        if self.fixed_ridge not in self.ridges or self.capacity < 0:
            raise ValueError('fixed ridge must be a candidate and capacity nonnegative')


def split_logs(data, fraction, seed):
    permutation = np.random.default_rng(seed).permutation(len(data.requests))
    k = int(len(permutation) * fraction)
    def subset(indices):
        return LoggedDataset(tuple(data.requests[int(i)] for i in indices),
                             tuple(data.events[int(i)] for i in indices))
    return subset(permutation[:k]), subset(permutation[k:])


def run(config: M45Config, progress=None):
    rows, choices = [], []
    projection = projection_matrix(config.capacity, config.seed + 777)
    total = config.trials * len(config.requests) * len(config.explorations)
    done = 0
    for n in config.requests:
        for exploration in config.explorations:
            for trial in range(config.trials):
                seed = config.seed + 1009 * trial
                sim = SyntheticConfig(n_requests=n, exploration=exploration, seed=seed)
                world = SyntheticWorld(sim)
                requests = generate_requests(sim)
                logging = lambda r: epsilon_greedy_probabilities(r, exploration)
                logs = log_interactions(world, requests, logging, seed=seed + 2)
                training, evaluation = split_logs(logs, config.train_fraction, seed + 11)
                # Select using training logs only; evaluation outcomes never enter selection.
                cv = [(ridge, brier_score(training, cross_fit(training, projection, ridge,
                       config.folds, seed + 3))) for ridge in config.ridges]
                selected = choose_by_brier(cv)
                # All strategies are scored on exactly the same held-out requests.
                models = {
                    'fixed': (config.fixed_ridge, fit(training, projection, config.fixed_ridge)),
                    'independent-brier': (selected, fit(training, projection, selected)),
                    'evaluation-oof': (selected, cross_fit(evaluation, projection, selected,
                                                          config.folds, seed + 5)),
                }
                truth = oracle_policy_value(world, evaluation.requests, uniform_probabilities,
                                            f'm45-{n}-{exploration}-{trial}', 'uniform').true_expected_reward
                choices.append(dict(requests=n, exploration=exploration, trial=trial,
                                    train_requests=len(training.requests), eval_requests=len(evaluation.requests),
                                    fixed_ridge=config.fixed_ridge, brier_ridge=selected))
                for strategy, (ridge, model) in models.items():
                    result = doubly_robust(evaluation, uniform_probabilities, model)
                    rows.append(dict(requests=n, exploration=exploration, trial=trial,
                                     strategy=strategy, selected_ridge=ridge,
                                     train_requests=len(training.requests), eval_requests=len(evaluation.requests),
                                     truth=truth, estimate=result.value, error=result.value-truth,
                                     squared_error=(result.value-truth)**2,
                                     brier=brier_score(evaluation, model),
                                     ess=result.effective_sample_size))
                done += 1
                if progress:
                    progress(done, total)
    return rows, choices


def paired_comparisons(rows):
    result = []
    for n, e in dict.fromkeys((r['requests'], r['exploration']) for r in rows):
        trials = {}
        for r in rows:
            if r['requests'] == n and r['exploration'] == e:
                trials.setdefault(r['trial'], {})[r['strategy']] = r['squared_error']
        for strategy in ('independent-brier', 'evaluation-oof'):
            delta = np.array([v[strategy] - v['fixed'] for v in trials.values()])
            mean = float(delta.mean())
            se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
            result.append(dict(requests=n, exploration=e, strategy=strategy, comparator='fixed',
                               trials=len(delta), mean_delta_mse=mean, ci_low=mean-1.96*se,
                               ci_high=mean+1.96*se, wins=int(np.sum(delta < 0))))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--requests', type=int, nargs='+', default=[100, 500])
    p.add_argument('--explorations', type=float, nargs='+', default=[0.05, 0.2])
    p.add_argument('--ridges', type=float, nargs='+', default=[0.1, 10, 100])
    p.add_argument('--fixed-ridge', type=float, default=10)
    p.add_argument('--capacity', type=int, default=40)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--train-fraction', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m45_independent'))
    a = p.parse_args(argv)
    config = M45Config(a.trials, tuple(a.requests), tuple(a.explorations), tuple(a.ridges),
                       a.fixed_ridge, a.capacity, a.folds, a.train_fraction, a.seed)
    rows, choices = run(config, lambda i, total: print(f'Completed {i}/{total}', flush=True))
    for name, values in [('summary.csv', summarize(rows)), ('trials.csv', rows),
                         ('paired.csv', paired_comparisons(rows)), ('choices.csv', choices)]:
        write_csv(a.output_dir / name, values)
    for row in summarize(rows):
        print(f"n={row['requests']} eps={row['exploration']:.2f} {row['strategy']:19s} "
              f"RMSE={row['rmse']:.5f} Brier={row['brier']:.5f} ESS={row['ess']:.1f}")
    print(f'Wrote {a.output_dir}')


if __name__ == '__main__':
    main()
