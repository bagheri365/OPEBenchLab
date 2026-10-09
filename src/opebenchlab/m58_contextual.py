"""M5.8: learn request-conditioned slate interactions from aggregate rewards.

The contextual model sees categorical request context and slate identities, but no
hand-coded equality-to-target feature. The oracle does see that equality feature.
This is a synthetic diagnostic, not a production OPE estimator.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m55_benchmark import features, fit_ridge
from opebenchlab.m56_diagnostics import targeted_features


@dataclass(frozen=True)
class M58Config:
    trials: int = 10
    requests: tuple[int, ...] = (500, 2000)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    candidates: int = 5
    k: int = 3
    pair_bonus: float = 0.2
    noise: float = 0.05
    exploration: float = 0.5
    train_fraction: float = 0.7
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or min(self.requests) < 10:
            raise ValueError('invalid trial/request count')
        if not 2 <= self.k <= self.candidates <= 7:
            raise ValueError('invalid slate dimensions')
        if not self.ridges or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('invalid ridges')
        if not np.isfinite(self.pair_bonus) or not 0 <= self.pair_bonus <= 1:
            raise ValueError('invalid bonus')
        if not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError('invalid noise')
        if not 0 < self.exploration <= 1 or not 0 < self.train_fraction < 1:
            raise ValueError('invalid exploration or train fraction')


def contextual_indices(slates, contexts, pairs):
    """Categorical context-pair x logged-pair interaction; no target-match test."""
    pair_index = {p: i for i, p in enumerate(pairs)}
    p = len(pairs)
    return np.asarray([pair_index[c] * p + pair_index[s[:2]]
                       for s, c in zip(slates, contexts)], dtype=int)


def fit_context_residual(indices, residuals, ridge, categories):
    """Ridge-shrunk categorical residuals, with zero as the prior mean."""
    counts = np.bincount(indices, minlength=categories)
    totals = np.bincount(indices, weights=residuals, minlength=categories)
    return totals / (counts + ridge)


def run(config: M58Config) -> list[dict]:
    items = tuple(f'item-{i}' for i in range(config.candidates))
    pairs = list(permutations(items, 2))
    slates = list(permutations(items, config.k))
    pair_to_slate = {p: next(s for s in slates if s[:2] == p) for p in pairs}
    # Every request context is an observable ordered item pair. The synthetic
    # reward rule (exact match) is used only for data generation and oracle.
    all_contexts = [p for p in pairs for _ in slates]
    all_slates = [s for _ in pairs for s in slates]
    xa = np.stack([features(s, items) for s in all_slates])
    xp = np.stack([features(s, items, True) for s in all_slates])
    xo = np.stack([targeted_features(s, c, items) for s, c in zip(all_slates, all_contexts)])
    ix = contextual_indices(all_slates, all_contexts, pairs)
    # Nonconstant item-position base reward, unknown to the fitted models.
    item_weights = np.linspace(-0.04, 0.04, config.candidates)
    item_idx = {item: i for i, item in enumerate(items)}
    base = np.array([0.3 + sum(item_weights[item_idx[item]] / (j + 1)
                                      for j, item in enumerate(s))
                     for s in all_slates])
    matches = np.array([float(s[:2] == c) for s, c in zip(all_slates, all_contexts)])
    truth = base + config.pair_bonus * matches
    # Target-policy population: all contexts, each with a deterministic target slate.
    target_idx = np.array([pairs.index(p) * len(slates) + slates.index(pair_to_slate[p])
                           for p in pairs])
    true_value = float(truth[target_idx].mean())
    # Context-uniform logging mixture: explore uniformly, otherwise select target.
    rng_size = len(pairs)
    rows = []
    for n in config.requests:
        for trial in range(config.trials):
            rng = np.random.default_rng(config.seed + n * 100003 + trial)
            context_ids = rng.integers(rng_size, size=n)
            logged_ids = rng.integers(len(slates), size=n)
            exploit = rng.random(n) >= config.exploration
            for i in np.flatnonzero(exploit):
                logged_ids[i] = slates.index(pair_to_slate[pairs[context_ids[i]]])
            selected = context_ids * len(slates) + logged_ids
            y = truth[selected] + rng.normal(0, config.noise, n)
            perm = rng.permutation(n)
            cut = max(2, min(n - 2, int(n * config.train_fraction)))
            train, test = selected[perm[:cut]], selected[perm[cut:]]
            yt, ye = y[perm[:cut]], y[perm[cut:]]
            for ridge in config.ridges:
                for name, matrix in [('additive', xa), ('generic-pairwise', xp),
                                     ('target-pair-oracle', xo)]:
                    beta = fit_ridge(matrix[train], yt, ridge)
                    pred_test = matrix[test] @ beta
                    pred_target = matrix[target_idx] @ beta
                    rows.append(dict(requests=n, trial=trial, ridge=ridge, model=name,
                                     train_size=len(train), test_size=len(test),
                                     train_matches=int(matches[train].sum()),
                                     true_target_value=true_value,
                                     heldout_rmse=float(np.sqrt(np.mean((pred_test - ye)**2))),
                                     target_estimate=float(pred_target.mean()),
                                     target_error=float(pred_target.mean() - true_value)))
                # Context x logged-pair residuals on top of an additive fit.
                # This is a two-stage training fit, not cross-fitting or DR.
                additive_beta = fit_ridge(xa[train], yt, ridge)
                residual = yt - xa[train] @ additive_beta
                coefficients = fit_context_residual(ix[train], residual, ridge, len(pairs)**2)
                pred_test = xa[test] @ additive_beta + coefficients[ix[test]]
                pred_target = xa[target_idx] @ additive_beta + coefficients[ix[target_idx]]
                rows.append(dict(requests=n, trial=trial, ridge=ridge, model='contextual-pairwise',
                                 train_size=len(train), test_size=len(test),
                                 train_matches=int(matches[train].sum()),
                                 true_target_value=true_value,
                                 heldout_rmse=float(np.sqrt(np.mean((pred_test - ye)**2))),
                                 target_estimate=float(pred_target.mean()),
                                 target_error=float(pred_target.mean() - true_value)))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row['requests'], row['ridge'], row['model']), []).append(row)
    return [dict(requests=n, ridge=ridge, model=model, trials=len(group),
                 true_target_value=group[0]['true_target_value'],
                 mean_train_matches=float(np.mean([r['train_matches'] for r in group])),
                 heldout_rmse=float(np.mean([r['heldout_rmse'] for r in group])),
                 target_bias=float(np.mean([r['target_error'] for r in group])),
                 target_rmse=float(np.sqrt(np.mean([r['target_error']**2 for r in group]))))
            for (n, ridge, model), group in sorted(groups.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    parser.add_argument('--ridges', type=float, nargs='+', default=[0.01, 1, 10])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--pair-bonus', type=float, default=0.2)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--exploration', type=float, default=0.5)
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m58_contextual'))
    args = parser.parse_args(argv)
    config = M58Config(args.trials, tuple(args.requests), tuple(args.ridges),
                       args.candidates, args.k, args.pair_bonus, args.noise,
                       args.exploration, args.train_fraction, args.seed)
    rows = run(config)
    write_csv(args.output_dir / 'trials.csv', rows)
    write_csv(args.output_dir / 'summary.csv', summarize(rows))
    print(f'Wrote {len(rows)} trial rows to {args.output_dir}')


if __name__ == '__main__':
    main()
