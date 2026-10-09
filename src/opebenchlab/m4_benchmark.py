"""M4.1 paired Monte Carlo study: nuisance fitting versus OPE accuracy.

Oracle values are used only after estimates have been computed, for scoring.
Brier scores are on logged actions; in-sample Brier is optimistic.
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opebenchlab.estimators import doubly_robust, ips
from opebenchlab.reward_models import brier_score, cross_fit_reward_model, fit_reward_model
from opebenchlab.synthetic import (SyntheticConfig, SyntheticWorld,
    epsilon_greedy_probabilities, generate_requests, log_interactions,
    oracle_policy_value, uniform_probabilities)


def fixed_reward_model(request, item_id):
    return 0.5


@dataclass(frozen=True)
class M4Config:
    trials: int = 30
    requests: int = 500
    explorations: tuple[float, ...] = (0.05, 0.2, 1.0)
    folds: int = 5
    seed: int = 42
    ridge: float = 1.0

    def __post_init__(self):
        if self.trials < 2 or self.requests < 2 or not 2 <= self.folds <= self.requests:
            raise ValueError('trials >= 2, requests >= 2 and 2 <= folds <= requests required')
        if not self.explorations or any(not np.isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('explorations must be nonempty and in (0, 1]')
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('ridge must be positive and finite')


def run(config: M4Config, progress=None) -> list[dict]:
    """Each estimator sees identical logs within an exploration/trial pair."""
    rows = []
    total = config.trials * len(config.explorations)
    done = 0
    for exploration in config.explorations:
        for trial in range(config.trials):
            seed = config.seed + trial * 1009
            sim = SyntheticConfig(n_requests=config.requests, exploration=exploration, seed=seed)
            world = SyntheticWorld(sim)
            requests = generate_requests(sim)
            logging = lambda request: epsilon_greedy_probabilities(request, exploration)
            data = log_interactions(world, requests, logging, seed=seed + 2)
            truth = oracle_policy_value(world, requests, uniform_probabilities,
                                        f'm4-{trial}', 'uniform').true_expected_reward
            fitted = fit_reward_model(data, ridge=config.ridge)
            cross_fitted = cross_fit_reward_model(data, folds=config.folds,
                                                  seed=seed + 3, ridge=config.ridge)
            methods = (
                ('IPS', ips(data, uniform_probabilities), float('nan')),
                ('DR-fixed', doubly_robust(data, uniform_probabilities, fixed_reward_model),
                 brier_score(data, fixed_reward_model)),
                ('DR-in-sample', doubly_robust(data, uniform_probabilities, fitted),
                 brier_score(data, fitted)),
                ('DR-cross-fitted', doubly_robust(data, uniform_probabilities, cross_fitted),
                 brier_score(data, cross_fitted)),
            )
            for name, estimate, brier in methods:
                rows.append(dict(exploration=exploration, trial=trial, estimator=name,
                                 estimate=estimate.value, truth=truth,
                                 error=estimate.value - truth, brier=brier,
                                 ess=estimate.effective_sample_size))
            done += 1
            if progress is not None:
                progress(done, total)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for exploration in dict.fromkeys(row['exploration'] for row in rows):
        for estimator in dict.fromkeys(row['estimator'] for row in rows):
            group = [r for r in rows if r['exploration'] == exploration and r['estimator'] == estimator]
            if not group:
                continue
            errors = np.asarray([r['error'] for r in group])
            briers = np.asarray([r['brier'] for r in group])
            result.append(dict(exploration=exploration, estimator=estimator, trials=len(group),
                               bias=float(errors.mean()), rmse=float(np.sqrt(np.mean(errors**2))),
                               std_error=float(errors.std(ddof=1)) if len(group) > 1 else float('nan'),
                               mean_brier=float(np.nanmean(briers)) if np.isfinite(briers).any() else float('nan'),
                               mean_ess=float(np.mean([r['ess'] for r in group]))))
    return result


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', type=int, default=500)
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.05, 0.2, 1.0])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m4_crossfit'))
    args = parser.parse_args(argv)
    config = M4Config(args.trials, args.requests, tuple(args.explorations), args.folds, args.seed, args.ridge)
    rows = run(config, progress=lambda done, total: print(f'Completed {done}/{total}', flush=True))
    summary = summarize(rows)
    write_csv(args.output_dir / 'trials.csv', rows)
    write_csv(args.output_dir / 'summary.csv', summary)
    print('\nExploration  Estimator         Bias       RMSE       Brier      ESS')
    for r in summary:
        print(f"{r['exploration']:11.2f}  {r['estimator']:16s} {r['bias']:9.4f}  {r['rmse']:9.4f}  {r['mean_brier']:9.4f}  {r['mean_ess']:7.1f}")
    print(f'Wrote {args.output_dir / "summary.csv"} and {args.output_dir / "trials.csv"}')


if __name__ == '__main__':
    main()
