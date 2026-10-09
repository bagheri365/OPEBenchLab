"""M5.4: position and pairwise OPE with explicitly decomposed rewards.

The component rewards (position clicks, pair bonus, triple bonus) are observed
separately. Pairwise estimators do NOT reweight the entire slate reward.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import isfinite
from pathlib import Path

import numpy as np

from opebenchlab.m51_slates import deterministic_top_k, expected_slate_reward, logging_scores, sample_slate
from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m53_benchmark import _estimate, position_marginals, summarize_m53
from opebenchlab.synthetic import SyntheticConfig, SyntheticWorld, generate_requests


@lru_cache(maxsize=512)
def _pair_marginals_cached(scores: tuple[float, ...], k: int) -> np.ndarray:
    """Exact P(A_i=a,A_j=b), for i<j, by enumerating ordered prefixes."""
    weights = np.asarray(scores, dtype=float)
    n = len(weights)
    if n < 2 or n > 10 or not 2 <= k <= n or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError('require positive finite scores and 2 <= K <= candidates <= 10')
    result = np.zeros((k, k, n, n))
    for slate in permutations(range(n), k):
        probability = 1.0
        remaining = float(weights.sum())
        for item in slate:
            probability *= weights[item] / remaining
            remaining -= weights[item]
        for i in range(k):
            for j in range(i + 1, k):
                result[i, j, slate[i], slate[j]] += probability
    return result


def pair_marginals(scores: np.ndarray, k: int) -> np.ndarray:
    """Return joint ordered-position marginals, shape (K,K,N,N)."""
    return _pair_marginals_cached(tuple(np.asarray(scores, dtype=float)), k).copy()


@dataclass(frozen=True)
class M54Config:
    trials: int = 30
    requests: tuple[int, ...] = (500, 2000)
    ks: tuple[int, ...] = (3, 4)
    explorations: tuple[float, ...] = (0.2, 1.0)
    scenarios: tuple[str, ...] = ('additive', 'pairwise', 'threeway')
    candidates: int = 5
    pair_bonus: float = 0.2
    threeway_bonus: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or not self.ks or not self.explorations or not self.scenarios or not 3 <= self.candidates <= 10:
            raise ValueError('invalid benchmark grid')
        if any(n < 1 for n in self.requests) or any(k < 3 or k > self.candidates for k in self.ks):
            raise ValueError('invalid requests or K')
        if any(not isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('exploration must be in (0, 1]')
        if set(self.scenarios) - {'additive', 'pairwise', 'threeway'}:
            raise ValueError('unknown scenario')
        if any(not isfinite(b) or not 0 <= b <= 0.5 for b in (self.pair_bonus, self.threeway_bonus)):
            raise ValueError('bonuses must be in [0, 0.5]')


def run(config: M54Config) -> list[dict]:
    rows = []
    for n in config.requests:
        for k in config.ks:
            for exploration in config.explorations:
                for trial in range(config.trials):
                    seed = config.seed + n * 100003 + k * 1009 + int(exploration * 1000) * 10000019 + trial
                    sc = SyntheticConfig(n_requests=n, candidates_per_request=config.candidates, exploration=exploration, seed=seed)
                    world = SyntheticWorld(sc)
                    rng = np.random.default_rng(seed + 517)
                    full_w, position_w, pair_w, clicks, pair_match, triple_match, truths = [], [], [], [], [], [], []
                    for request in generate_requests(sc):
                        target = deterministic_top_k(request, k)
                        scores = logging_scores(request, exploration)
                        slate, conditional = sample_slate(request, scores, k, rng)
                        observed_clicks = np.array([float(rng.random() < world.click_probability(request, item) / (j + 1)) / k for j, item in enumerate(slate)])
                        pm = position_marginals(scores, k)
                        jm = pair_marginals(scores, k)
                        pw = np.array([1 / pm[j, request.candidate_item_ids.index(slate[j])] if slate[j] == target[j] else 0.0 for j in range(k)])
                        a, b = (request.candidate_item_ids.index(item) for item in target[:2])
                        joint_weight = 1 / jm[0, 1, a, b] if slate[:2] == target[:2] else 0.0
                        full_w.append(1 / np.prod(conditional) if slate == target else 0.0)
                        position_w.append(pw)
                        pair_w.append(joint_weight)
                        clicks.append(observed_clicks)
                        pair_match.append(float(slate[:2] == target[:2]))
                        triple_match.append(float(slate[:3] == target[:3]))
                        truths.append(expected_slate_reward(world, request, target))
                    fw, pw, jw = np.asarray(full_w), np.asarray(position_w), np.asarray(pair_w)
                    parts, pmatch, tmatch = np.asarray(clicks), np.asarray(pair_match), np.asarray(triple_match)
                    for scenario in config.scenarios:
                        pb = config.pair_bonus if scenario in ('pairwise', 'threeway') else 0.0
                        tb = config.threeway_bonus if scenario == 'threeway' else 0.0
                        reward = parts.sum(axis=1) + pb * pmatch + tb * tmatch
                        oracle = float(np.mean(truths)) + pb + tb
                        estimates = {'Slate IPS': _estimate(fw, reward, False), 'Slate SNIPS': _estimate(fw, reward, True)}
                        for normalized, suffix in ((False, 'IPS'), (True, 'SNIPS')):
                            pos_results = [_estimate(pw[:, j], parts[:, j], normalized) for j in range(k)]
                            pos_value = float(sum(v for v, _ in pos_results))
                            pos_ess = float(np.mean([ess for _, ess in pos_results]))
                            # Only position-specific clicks are identifiable from the
                            # additive decomposition. Pair bonus is a distinct observed
                            # component and must be weighted with its joint propensity.
                            pair_value, joint_ess = _estimate(jw, pb * pmatch, normalized)
                            estimates[f'Position {suffix}'] = (pos_value, pos_ess)
                            estimates[f'Pairwise {suffix}'] = (pos_value + pair_value, min(pos_ess, joint_ess))
                        for name, (value, ess) in estimates.items():
                            defined = bool(np.isfinite(value))
                            rows.append(dict(requests=n, k=k, exploration=exploration, scenario=scenario,
                                             trial=trial, estimator=name, oracle=oracle, estimate=value,
                                             error=value - oracle if defined else float('nan'),
                                             squared_error=(value - oracle)**2 if defined else float('nan'),
                                             ess=ess, full_matches=int(np.count_nonzero(fw)),
                                             pair_matches=int(np.count_nonzero(jw)), undefined=int(not defined)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    parser.add_argument('--ks', type=int, nargs='+', default=[3, 4])
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.2, 1.0])
    parser.add_argument('--scenarios', nargs='+', default=['additive', 'pairwise', 'threeway'])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--pair-bonus', type=float, default=0.2)
    parser.add_argument('--threeway-bonus', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m54_pairwise'))
    args = parser.parse_args(argv)
    cfg = M54Config(args.trials, tuple(args.requests), tuple(args.ks), tuple(args.explorations), tuple(args.scenarios), args.candidates, args.pair_bonus, args.threeway_bonus, args.seed)
    rows = run(cfg)
    summary = summarize_m53(rows)
    write_csv(args.output_dir / 'trials.csv', rows)
    write_csv(args.output_dir / 'summary.csv', summary)
    for row in summary:
        print(f"{row['requests']:5d} K={row['k']} eps={row['exploration']:.2f} {row['scenario']:9s} {row['estimator']:15s} RMSE={row['rmse']:.4f} ESS={row['mean_ess']:.1f}")
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
