"""M4.2: paired reward-model capacity and cross-fitting experiments.

Oracle values are used for scoring only, never for model fitting.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opebenchlab.estimators import doubly_robust
from opebenchlab.m4_benchmark import write_csv
from opebenchlab.reward_models import pre_action_features, brier_score
from opebenchlab.synthetic import (SyntheticConfig, SyntheticWorld,
    epsilon_greedy_probabilities, generate_requests, log_interactions,
    oracle_policy_value, uniform_probabilities)


@dataclass(frozen=True)
class FeatureModel:
    coefficients: np.ndarray
    projection: np.ndarray | None

    def __call__(self, request, item_id):
        x = pre_action_features(request, item_id)
        if self.projection is not None:
            x = np.concatenate((x, np.tanh(x[1:] @ self.projection)))
        return float(np.clip(x @ self.coefficients, 0, 1))


def projection_matrix(capacity: int, seed: int) -> np.ndarray | None:
    if capacity < 0:
        raise ValueError('capacity must be nonnegative')
    if capacity == 0:
        return None
    return np.random.default_rng(seed).normal(size=(5, capacity))


def fit(data, projection, ridge: float):
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be positive and finite')
    if not data.events:
        raise ValueError('training data must be nonempty')
    features = []
    for request, event in zip(data.requests, data.events):
        x = pre_action_features(request, event.item_id)
        if projection is not None:
            x = np.concatenate((x, np.tanh(x[1:] @ projection)))
        features.append(x)
    x = np.stack(features)
    y = np.array([event.reward for event in data.events], dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return FeatureModel(coefficients, projection)


@dataclass(frozen=True)
class FoldModel:
    models: tuple[FeatureModel, ...]
    assignment: dict[str, int]

    def __call__(self, request, item_id):
        return self.models[self.assignment[request.request_id]](request, item_id)


def cross_fit(data, projection, ridge: float, folds: int, seed: int):
    from opebenchlab.synthetic import LoggedDataset
    n = len(data.requests)
    if not 2 <= folds <= n:
        raise ValueError('invalid folds')
    permutation = np.random.default_rng(seed).permutation(n)
    models, assignment = [], {}
    for fold, heldout in enumerate(np.array_split(permutation, folds)):
        excluded = set(int(i) for i in heldout)
        train = LoggedDataset(tuple(r for i, r in enumerate(data.requests) if i not in excluded),
                              tuple(e for i, e in enumerate(data.events) if i not in excluded))
        models.append(fit(train, projection, ridge))
        for i in heldout:
            assignment[data.requests[int(i)].request_id] = fold
    return FoldModel(tuple(models), assignment)


@dataclass(frozen=True)
class M42Config:
    trials: int = 30
    requests: tuple[int, ...] = (100, 500)
    explorations: tuple[float, ...] = (0.05, 0.2)
    capacities: tuple[int, ...] = (0, 40)
    folds: int = 5
    ridge: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if self.trials < 2 or not self.requests or min(self.requests) < self.folds or self.folds < 2:
            raise ValueError('trials >= 2 and requests >= folds >= 2 required')
        if not self.explorations or any(not 0 < e <= 1 for e in self.explorations):
            raise ValueError('explorations must be in (0, 1]')
        if not self.capacities or any(c < 0 for c in self.capacities):
            raise ValueError('capacities must be nonnegative')
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('ridge must be positive')


def run(config: M42Config, progress=None):
    rows = []
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
                truth = oracle_policy_value(world, requests, uniform_probabilities,
                                            f'm42-{n}-{trial}', 'uniform').true_expected_reward
                for capacity in config.capacities:
                    projection = projection_matrix(capacity, config.seed + 777)
                    models = (('in-sample', fit(data, projection, config.ridge)),
                              ('cross-fitted', cross_fit(data, projection, config.ridge,
                                                        config.folds, seed + 3)))
                    for strategy, model in models:
                        estimate = doubly_robust(data, uniform_probabilities, model)
                        rows.append(dict(requests=n, exploration=exploration, trial=trial,
                                         capacity=capacity, strategy=strategy, truth=truth,
                                         estimate=estimate.value, error=estimate.value-truth,
                                         squared_error=(estimate.value-truth)**2,
                                         brier=brier_score(data, model),
                                         ess=estimate.effective_sample_size))
                done += 1
                if progress:
                    progress(done, total)
    return rows


def summarize(rows):
    summary = []
    keys = dict.fromkeys((r['requests'], r['exploration'], r['capacity'], r['strategy']) for r in rows)
    for n, exploration, capacity, strategy in keys:
        group = [r for r in rows if (r['requests'], r['exploration'], r['capacity'], r['strategy']) ==
                 (n, exploration, capacity, strategy)]
        errors = np.array([r['error'] for r in group])
        summary.append(dict(requests=n, exploration=exploration, capacity=capacity,
                            strategy=strategy, trials=len(group), bias=float(errors.mean()),
                            rmse=float(np.sqrt(np.mean(errors**2))),
                            brier=float(np.mean([r['brier'] for r in group])),
                            ess=float(np.mean([r['ess'] for r in group]))))
    return summary


def paired_differences(rows):
    """Cross-fitted minus in-sample squared error, paired on identical logs."""
    result = []
    keys = dict.fromkeys((r['requests'], r['exploration'], r['capacity']) for r in rows)
    for n, exploration, capacity in keys:
        group = [r for r in rows if (r['requests'], r['exploration'], r['capacity']) ==
                 (n, exploration, capacity)]
        by_trial = {}
        for r in group:
            by_trial.setdefault(r['trial'], {})[r['strategy']] = r['squared_error']
        differences = np.array([v['cross-fitted'] - v['in-sample'] for v in by_trial.values()])
        # Normal approximation, descriptive Monte Carlo interval; paired by trial.
        se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
        mean = float(differences.mean())
        result.append(dict(requests=n, exploration=exploration, capacity=capacity,
                           trials=len(differences), mean_delta_mse=mean,
                           ci_low=mean-1.96*se, ci_high=mean+1.96*se,
                           crossfit_wins=int(np.sum(differences < 0))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', type=int, nargs='+', default=[100, 500])
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.05, 0.2])
    parser.add_argument('--capacities', type=int, nargs='+', default=[0, 40])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--ridge', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m42_capacity'))
    args = parser.parse_args(argv)
    config = M42Config(args.trials, tuple(args.requests), tuple(args.explorations),
                       tuple(args.capacities), args.folds, args.ridge, args.seed)
    rows = run(config, lambda done, total: print(f'Completed {done}/{total}', flush=True))
    summary = summarize(rows)
    paired = paired_differences(rows)
    for filename, values in [('trials.csv', rows), ('summary.csv', summary), ('paired.csv', paired)]:
        write_csv(args.output_dir / filename, values)
    for r in paired:
        print(f"n={r['requests']} eps={r['exploration']:.2f} capacity={r['capacity']}: "
              f"delta MSE={r['mean_delta_mse']:+.6f} "
              f"95% approx CI=[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]")
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
