"""M5.6: diagnose learnability of target-specific pair rewards from aggregate logs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opebenchlab.m51_slates import deterministic_top_k, logging_scores, sample_slate
from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m55_benchmark import features, fit_ridge
from opebenchlab.synthetic import SyntheticConfig, SyntheticWorld, generate_requests


@dataclass(frozen=True)
class M56Config:
    trials: int = 10
    requests: tuple[int, ...] = (500, 2000)
    explorations: tuple[float, ...] = (0.2, 1.0)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    k: int = 3
    candidates: int = 5
    pair_bonus: float = 0.2
    train_fraction: float = 0.5
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or any(n < 4 for n in self.requests):
            raise ValueError('invalid requests/trials')
        if not 2 <= self.k <= self.candidates <= 10:
            raise ValueError('invalid slate size/candidates')
        if not self.explorations or any(not np.isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('invalid exploration')
        if not self.ridges or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('invalid ridge')
        if not np.isfinite(self.pair_bonus) or not 0 <= self.pair_bonus <= 0.5:
            raise ValueError('invalid bonus')
        if not 0 < self.train_fraction < 1:
            raise ValueError('invalid train fraction')


def targeted_features(slate, target, candidates):
    """Additive indicators plus a known target-pair match indicator.

    This is an intentionally privileged diagnostic feature: the target ranking
    is available to the experimenter, but the reward-generating rule is not
    assumed known in real deployments.
    """
    return np.r_[features(slate, candidates), float(slate[:2] == target[:2])]


def coefficient_recovery(ridge=1e-6, bonus=0.2):
    """Noise-free balanced experiment; isolates design/regularization from overlap."""
    from itertools import permutations
    candidates = tuple(f'item-{i}' for i in range(5))
    slates = list(permutations(candidates, 3))
    target = slates[0]
    x = np.stack([targeted_features(s, target, candidates) for s in slates])
    y = x[:, -1] * bonus
    beta = fit_ridge(x, y, ridge)
    return float(beta[-1]), int(np.linalg.matrix_rank(x)), x.shape[1]


def run(config: M56Config):
    rows = []
    for n in config.requests:
        for exploration in config.explorations:
            for trial in range(config.trials):
                seed = config.seed + n * 100003 + int(exploration * 1000) * 10000019 + trial
                sc = SyntheticConfig(n_requests=n, candidates_per_request=config.candidates,
                                     exploration=exploration, seed=seed)
                world = SyntheticWorld(sc)
                rng = np.random.default_rng(seed + 517)
                records = []
                for request in generate_requests(sc):
                    target = deterministic_top_k(request, config.k)
                    slate, _ = sample_slate(request, logging_scores(request, exploration), config.k, rng)
                    base = sum(float(rng.random() < world.click_probability(request, item) / (j + 1))
                               for j, item in enumerate(slate)) / config.k
                    match = int(slate[:2] == target[:2])
                    records.append((request, slate, target, base + config.pair_bonus * match, match))
                perm = np.random.default_rng(seed + 911).permutation(n)
                split = max(2, min(n - 2, int(n * config.train_fraction)))
                train = [records[i] for i in perm[:split]]
                evaluation = [records[i] for i in perm[split:]]
                y = np.array([r[3] for r in train])
                matches = sum(r[4] for r in train)
                for ridge in config.ridges:
                    for name in ('additive', 'pairwise', 'target-pair'):
                        def design(record):
                            request, slate, target = record[:3]
                            if name == 'target-pair':
                                return targeted_features(slate, target, request.candidate_item_ids)
                            return features(slate, request.candidate_item_ids, name == 'pairwise')
                        x = np.stack([design(r) for r in train])
                        xt = np.stack([design((r[0], r[2], r[2])) for r in evaluation])
                        beta = fit_ridge(x, y, ridge)
                        # Contrast against a baseline fit to the same click noise but no bonus.
                        baseline = fit_ridge(x, np.array([r[3] - config.pair_bonus * r[4] for r in train]), ridge)
                        interaction_effect = float(np.mean(xt @ (beta - baseline)))
                        rows.append(dict(requests=n, exploration=exploration, trial=trial, ridge=ridge,
                                         model=name, train_requests=len(train), eval_requests=len(evaluation),
                                         pair_matches=matches, pair_match_rate=matches / len(train),
                                         design_rank=int(np.linalg.matrix_rank(x)), features=x.shape[1],
                                         interaction_effect=interaction_effect,
                                         interaction_error=interaction_effect - config.pair_bonus,
                                         recovered_coefficient=float(beta[-1] - baseline[-1]) if name == 'target-pair' else float('nan')))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['requests'], row['exploration'], row['ridge'], row['model'])
        groups.setdefault(key, []).append(row)
    result = []
    for (n, exploration, ridge, model), group in sorted(groups.items()):
        errors = np.array([r['interaction_error'] for r in group])
        result.append(dict(requests=n, exploration=exploration, ridge=ridge, model=model,
                           trials=len(group), mean_pair_matches=float(np.mean([r['pair_matches'] for r in group])),
                           mean_design_rank=float(np.mean([r['design_rank'] for r in group])),
                           mean_interaction_effect=float(np.mean([r['interaction_effect'] for r in group])),
                           interaction_bias=float(np.mean(errors)), interaction_rmse=float(np.sqrt(np.mean(errors**2))),
                           mean_recovered_coefficient=float(np.nanmean([r['recovered_coefficient'] for r in group]))
                           if model == 'target-pair' else float('nan')))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    p.add_argument('--explorations', type=float, nargs='+', default=[0.2, 1.0])
    p.add_argument('--ridges', type=float, nargs='+', default=[0.01, 1.0, 10.0])
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--candidates', type=int, default=5)
    p.add_argument('--pair-bonus', type=float, default=0.2)
    p.add_argument('--train-fraction', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m56_diagnostics'))
    a = p.parse_args(argv)
    config = M56Config(a.trials, tuple(a.requests), tuple(a.explorations), tuple(a.ridges),
                       a.k, a.candidates, a.pair_bonus, a.train_fraction, a.seed)
    rows = run(config)
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summarize(rows))
    recovered, rank, columns = coefficient_recovery()
    print(f'Balanced noise-free recovery: coefficient={recovered:.6f}, design rank={rank}/{columns}')
    print(f'Wrote {a.output_dir} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
