"""M5.5: learn slate reward structure from aggregate feedback only.

Models see no position clicks or hidden interaction components. The pairwise
model adds interactions between positions 0 and 1; it does not represent
arbitrary higher-order interactions. Independent training/evaluation splits
are by request, and all estimators use the same held-out logs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from opebenchlab.m51_slates import deterministic_top_k, expected_slate_reward, logging_scores, sample_slate
from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m53_benchmark import summarize_m53
from opebenchlab.synthetic import SyntheticConfig, SyntheticWorld, generate_requests


def features(slate: tuple[str, ...], candidates: tuple[str, ...], pairwise: bool = False) -> np.ndarray:
    """Intercept, item-at-position indicators, optionally first-two joint indicators."""
    k, c = len(slate), len(candidates)
    if not 2 <= k <= c or len(set(slate)) != k or any(item not in candidates for item in slate):
        raise ValueError('invalid slate')
    vector = np.zeros(1 + k * c + (c * c if pairwise else 0), dtype=float)
    vector[0] = 1.0
    indices = [candidates.index(item) for item in slate]
    for j, index in enumerate(indices):
        vector[1 + j * c + index] = 1.0
    if pairwise:
        vector[1 + k * c + indices[0] * c + indices[1]] = 1.0
    return vector


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Fit ridge with an unpenalized intercept, without seeing oracle values."""
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not isfinite(ridge) or ridge <= 0:
        raise ValueError('invalid ridge regression inputs')
    gram = x.T @ x
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0
    return np.linalg.solve(gram + penalty, x.T @ y)


@dataclass(frozen=True)
class M55Config:
    trials: int = 30
    requests: tuple[int, ...] = (500, 2000)
    ks: tuple[int, ...] = (3, 4)
    explorations: tuple[float, ...] = (0.2, 1.0)
    scenarios: tuple[str, ...] = ('additive', 'pairwise', 'threeway')
    candidates: int = 5
    ridge: float = 10.0
    train_fraction: float = 0.5
    pair_bonus: float = 0.2
    threeway_bonus: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or not self.ks or not self.explorations or not self.scenarios:
            raise ValueError('empty grid or invalid trials')
        if not 3 <= self.candidates <= 10 or any(k < 3 or k > self.candidates for k in self.ks):
            raise ValueError('invalid candidates or K')
        if any(n < 4 for n in self.requests) or not 0 < self.train_fraction < 1:
            raise ValueError('need training and evaluation requests')
        if any(not isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('invalid exploration')
        if set(self.scenarios) - {'additive', 'pairwise', 'threeway'}:
            raise ValueError('unknown scenario')
        if not isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError('ridge must be positive')
        if any(not isfinite(b) or not 0 <= b <= 0.5 for b in (self.pair_bonus, self.threeway_bonus)):
            raise ValueError('invalid interaction bonus')


def run(config: M55Config) -> list[dict]:
    rows = []
    for n in config.requests:
        for k in config.ks:
            for exploration in config.explorations:
                for trial in range(config.trials):
                    seed = config.seed + n * 100003 + k * 1009 + int(exploration * 1000) * 10000019 + trial
                    sc = SyntheticConfig(n_requests=n, candidates_per_request=config.candidates, exploration=exploration, seed=seed)
                    world = SyntheticWorld(sc)
                    rng = np.random.default_rng(seed + 517)
                    records = []
                    for request in generate_requests(sc):
                        target = deterministic_top_k(request, k)
                        slate, conditional = sample_slate(request, logging_scores(request, exploration), k, rng)
                        # These components are used only to create one aggregate observed reward.
                        click_total = sum(float(rng.random() < world.click_probability(request, item) / (j + 1)) for j, item in enumerate(slate)) / k
                        pair_match = float(slate[:2] == target[:2])
                        triple_match = float(slate[:3] == target[:3])
                        records.append((request, slate, target, click_total, pair_match, triple_match,
                                        float(np.prod(conditional)), expected_slate_reward(world, request, target)))
                    permutation = np.random.default_rng(seed + 911).permutation(n)
                    n_train = max(2, min(n - 2, int(n * config.train_fraction)))
                    train = [records[i] for i in permutation[:n_train]]
                    evaluation = [records[i] for i in permutation[n_train:]]
                    for scenario in config.scenarios:
                        pb = config.pair_bonus if scenario in ('pairwise', 'threeway') else 0.0
                        tb = config.threeway_bonus if scenario == 'threeway' else 0.0
                        def aggregate(r):
                            return r[3] + pb * r[4] + tb * r[5]
                        # Only aggregate observed reward and slate/context features enter training.
                        models = {}
                        for name, use_pairs in (('Additive DM', False), ('Pairwise DM', True)):
                            x = np.stack([features(r[1], r[0].candidate_item_ids, use_pairs) for r in train])
                            y = np.array([aggregate(r) for r in train])
                            beta = fit_ridge(x, y, config.ridge)
                            target_x = np.stack([features(r[2], r[0].candidate_item_ids, use_pairs) for r in evaluation])
                            models[name] = (float(np.mean(target_x @ beta)), float(len(evaluation)))
                        rewards = np.array([aggregate(r) for r in evaluation])
                        weights = np.array([1 / r[6] if r[1] == r[2] else 0.0 for r in evaluation])
                        ess = float(weights.sum() ** 2 / np.dot(weights, weights)) if np.any(weights) else 0.0
                        models['Slate IPS'] = (float(np.mean(weights * rewards)), ess)
                        models['Slate SNIPS'] = (float(np.dot(weights, rewards) / weights.sum()) if weights.sum() else float('nan'), ess)
                        oracle = float(np.mean([r[7] for r in evaluation])) + pb + tb
                        for name, (estimate, diagnostic) in models.items():
                            defined = bool(isfinite(estimate))
                            rows.append(dict(requests=n, k=k, exploration=exploration, scenario=scenario,
                                             trial=trial, estimator=name, oracle=oracle, estimate=estimate,
                                             error=estimate - oracle if defined else float('nan'),
                                             squared_error=(estimate - oracle) ** 2 if defined else float('nan'),
                                             ess=diagnostic, full_matches=int(np.count_nonzero(weights)),
                                             train_requests=len(train), eval_requests=len(evaluation),
                                             undefined=int(not defined)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    parser.add_argument('--ks', type=int, nargs='+', default=[3, 4])
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.2, 1.0])
    parser.add_argument('--scenarios', nargs='+', default=['additive', 'pairwise', 'threeway'])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--ridge', type=float, default=10)
    parser.add_argument('--train-fraction', type=float, default=0.5)
    parser.add_argument('--pair-bonus', type=float, default=0.2)
    parser.add_argument('--threeway-bonus', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m55_aggregate'))
    a = parser.parse_args(argv)
    config = M55Config(a.trials, tuple(a.requests), tuple(a.ks), tuple(a.explorations), tuple(a.scenarios),
                       a.candidates, a.ridge, a.train_fraction, a.pair_bonus, a.threeway_bonus, a.seed)
    rows = run(config)
    summary = summarize_m53(rows)
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summary)
    for row in summary:
        print(f"{row['requests']:5d} K={row['k']} eps={row['exploration']:.2f} {row['scenario']:9s} {row['estimator']:13s} RMSE={row['rmse']:.4f} ESS={row['mean_ess']:.1f}")
    print(f'Wrote {a.output_dir}')


if __name__ == '__main__':
    main()
