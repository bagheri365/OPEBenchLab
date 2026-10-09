"""M5.3: position-marginal slate IPS/SNIPS and additive-vs-interaction stress test.

Position estimators are unbiased for additive per-position expected rewards when
position marginals are correct. They are generally biased with slate interactions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np

from opebenchlab.m51_slates import (deterministic_top_k, expected_slate_reward,
                                    logging_scores, sample_slate)
from opebenchlab.m52_benchmark import write_csv
from opebenchlab.synthetic import SyntheticConfig, SyntheticWorld, generate_requests


@lru_cache(maxsize=512)
def _cached_marginals(weights: tuple[float, ...], k: int) -> np.ndarray:
    return _position_marginals(np.asarray(weights), k)


def position_marginals(scores: np.ndarray, k: int) -> np.ndarray:
    """Exact marginal probabilities; cached for repeated score vectors."""
    return _cached_marginals(tuple(np.asarray(scores, dtype=float)), k).copy()


def _position_marginals(scores: np.ndarray, k: int) -> np.ndarray:
    """Exact Plackett-Luce P(item i at position j), via subset dynamic programming.

    Exponential in candidate count: intended for small slate candidate pools.
    """
    weights = np.asarray(scores, dtype=float)
    n = len(weights)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('scores must be a nonempty vector of positive finite weights')
    if not 1 <= k <= n or n > 16:
        raise ValueError('require 1 <= K <= candidates <= 16')
    weights = weights / weights.sum()
    states = {0: 1.0}
    result = np.zeros((k, n))
    for j in range(k):
        nxt = {}
        for mask, mass in states.items():
            remaining = 1.0 - sum(weights[i] for i in range(n) if mask & (1 << i))
            for i in range(n):
                if not mask & (1 << i):
                    probability = mass * weights[i] / remaining
                    result[j, i] += probability
                    newmask = mask | (1 << i)
                    nxt[newmask] = nxt.get(newmask, 0.0) + probability
        states = nxt
    return result


@dataclass(frozen=True)
class M53Config:
    trials: int = 30
    requests: tuple[int, ...] = (500, 2000)
    ks: tuple[int, ...] = (2, 3, 4)
    explorations: tuple[float, ...] = (0.2, 1.0)
    scenarios: tuple[str, ...] = ('additive', 'interaction')
    candidates: int = 5
    interaction_bonus: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or self.candidates < 2 or self.candidates > 16 or not self.requests or not self.ks or not self.explorations or not self.scenarios:
            raise ValueError('invalid or empty benchmark grid')
        if any(n < 1 for n in self.requests) or any(k < 2 or k > self.candidates for k in self.ks):
            raise ValueError('invalid requests or K')
        if any(not isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('exploration must be in (0, 1]')
        if set(self.scenarios) - {'additive', 'interaction'}:
            raise ValueError('unknown scenario')
        if not isfinite(self.interaction_bonus) or not 0 <= self.interaction_bonus <= 0.5:
            raise ValueError('interaction bonus must be in [0, 0.5]')


def _estimate(weights: np.ndarray, rewards: np.ndarray, normalized: bool) -> tuple[float, float]:
    denominator = float(weights.sum()) if normalized else len(weights)
    squared = float(np.dot(weights, weights))
    ess = float(weights.sum()) ** 2 / squared if squared else 0.0
    return (float(np.dot(weights, rewards) / denominator) if denominator else float('nan'), ess)


def run(config: M53Config) -> list[dict]:
    rows = []
    for n in config.requests:
        for k in config.ks:
            for exploration in config.explorations:
                for trial in range(config.trials):
                    seed = config.seed + 100003 * n + 1009 * k + 10000019 * int(exploration * 1000) + trial
                    sc = SyntheticConfig(n_requests=n, candidates_per_request=config.candidates, exploration=exploration, seed=seed)
                    world = SyntheticWorld(sc)
                    requests = generate_requests(sc)
                    rng = np.random.default_rng(seed + 517)
                    # Same requests, logged slates, and Bernoulli clicks in both scenarios.
                    full_weights, full_additive, pos_weights, pos_additive, interactions, oracle_additive = [], [], [], [], [], []
                    for request in requests:
                        target = deterministic_top_k(request, k)
                        slate, conditional = sample_slate(request, logging_scores(request, exploration), k, rng)
                        clicks = np.array([float(rng.random() < world.click_probability(request, item) / (j + 1)) / k for j, item in enumerate(slate)])
                        scores = logging_scores(request, exploration)
                        marginals = position_marginals(scores, k)
                        weights = np.array([1.0 / marginals[j, request.candidate_item_ids.index(slate[j])] if slate[j] == target[j] else 0.0 for j in range(k)])
                        full_weights.append(1.0 / np.prod(conditional) if slate == target else 0.0)
                        full_additive.append(float(clicks.sum()))
                        pos_weights.append(weights)
                        pos_additive.append(clicks)
                        interactions.append(float(slate[:2] == target[:2]))
                        oracle_additive.append(expected_slate_reward(world, request, target))
                    fw = np.asarray(full_weights)
                    pw = np.asarray(pos_weights)
                    base = np.asarray(full_additive)
                    parts = np.asarray(pos_additive)
                    pair = np.asarray(interactions)
                    for scenario in config.scenarios:
                        bonus = config.interaction_bonus if scenario == 'interaction' else 0.0
                        # Interaction reward is credited to position 0, violating additivity.
                        observed_parts = parts.copy()
                        observed_parts[:, 0] += bonus * pair
                        rewards = base + bonus * pair
                        oracle = float(np.mean(oracle_additive)) + bonus
                        estimates = {}
                        estimates['Slate IPS'] = _estimate(fw, rewards, False)
                        estimates['Slate SNIPS'] = _estimate(fw, rewards, True)
                        ips_values = []
                        snips_values = []
                        for j in range(k):
                            ips_values.append(_estimate(pw[:, j], observed_parts[:, j], False)[0])
                            snips_values.append(_estimate(pw[:, j], observed_parts[:, j], True)[0])
                        estimates['Position IPS'] = (float(sum(ips_values)), float(np.mean([_estimate(pw[:, j], observed_parts[:, j], False)[1] for j in range(k)])))
                        estimates['Position SNIPS'] = (float(sum(snips_values)), float(np.mean([_estimate(pw[:, j], observed_parts[:, j], True)[1] for j in range(k)])))
                        for name, (value, ess) in estimates.items():
                            defined = bool(np.isfinite(value))
                            rows.append(dict(requests=n, k=k, exploration=exploration, scenario=scenario, trial=trial,
                                             estimator=name, oracle=oracle, estimate=value,
                                             error=value - oracle if defined else float('nan'),
                                             squared_error=(value - oracle) ** 2 if defined else float('nan'),
                                             ess=ess, full_matches=int(np.count_nonzero(fw)),
                                             undefined=int(not defined)))
    return rows


def summarize_m53(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(r['requests'], r['k'], r['exploration'], r['scenario'], r['estimator']) for r in rows})
    for key in keys:
        group = [r for r in rows if (r['requests'], r['k'], r['exploration'], r['scenario'], r['estimator']) == key]
        valid = [r for r in group if not r['undefined']]
        result.append(dict(requests=key[0], k=key[1], exploration=key[2], scenario=key[3], estimator=key[4],
                           trials=len(group), defined_trials=len(valid),
                           bias=float(np.mean([r['error'] for r in valid])) if valid else float('nan'),
                           rmse=float(np.sqrt(np.mean([r['squared_error'] for r in valid]))) if valid else float('nan'),
                           mean_ess=float(np.mean([r['ess'] for r in group])),
                           mean_full_matches=float(np.mean([r['full_matches'] for r in group])),
                           undefined_frequency=float(np.mean([r['undefined'] for r in group]))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', nargs='+', type=int, default=[500, 2000])
    parser.add_argument('--ks', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--explorations', nargs='+', type=float, default=[0.2, 1.0])
    parser.add_argument('--scenarios', nargs='+', default=['additive', 'interaction'])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--interaction-bonus', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m53_position'))
    args = parser.parse_args(argv)
    config = M53Config(args.trials, tuple(args.requests), tuple(args.ks), tuple(args.explorations), tuple(args.scenarios), args.candidates, args.interaction_bonus, args.seed)
    rows = run(config)
    summary = summarize_m53(rows)
    write_csv(args.output_dir / 'trials.csv', rows)
    write_csv(args.output_dir / 'summary.csv', summary)
    for row in summary:
        print(f"{row['requests']:5d} K={row['k']} eps={row['exploration']:.2f} {row['scenario']:11s} {row['estimator']:15s} RMSE={row['rmse']:.4f} ESS={row['mean_ess']:.1f} undefined={row['undefined_frequency']:.2f}")
    print(f'Wrote {args.output_dir / "summary.csv"} and {args.output_dir / "trials.csv"}')


if __name__ == '__main__':
    main()
