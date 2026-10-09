"""M5.9: cross-fitted contextual full-slate off-policy evaluation.

Synthetic logging has exact propensities. Ridge is selected on disjoint
validation requests; the oracle reward rule is used only to score estimates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m55_benchmark import features, fit_ridge
from opebenchlab.m58_contextual import contextual_indices, fit_context_residual


@dataclass(frozen=True)
class M59Config:
    trials: int = 10
    requests: tuple[int, ...] = (500, 2000)
    explorations: tuple[float, ...] = (0.2, 1.0)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    candidates: int = 5
    k: int = 3
    pair_bonus: float = 0.2
    noise: float = 0.05
    folds: int = 2
    validation_fraction: float = 0.2
    clip: float = 5.0
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or min(self.requests) < 20:
            raise ValueError('invalid trials or requests')
        if not 2 <= self.k <= self.candidates <= 7 or self.folds < 2:
            raise ValueError('invalid slate dimensions or folds')
        if not self.explorations or any(not np.isfinite(e) or not 0 < e <= 1 for e in self.explorations):
            raise ValueError('exploration must be in (0, 1]')
        if not self.ridges or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('ridges must be positive')
        if not 0 < self.validation_fraction < 0.5 or not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError('invalid validation fraction or clip')
        if not 0 <= self.pair_bonus <= 1 or self.noise < 0:
            raise ValueError('invalid reward parameters')
        if min(self.requests) * (1 - self.validation_fraction) < 2 * self.folds:
            raise ValueError('not enough evaluation requests for folds')


def fit_model(x, ix, y, train, ridge, categories, contextual):
    beta = fit_ridge(x[train], y[train], ridge)
    if not contextual:
        return beta, None
    residual = y[train] - x[train] @ beta
    return beta, fit_context_residual(ix[train], residual, ridge, categories)


def predict(model, x, ix, indices):
    beta, residual = model
    values = x[indices] @ beta
    return values if residual is None else values + residual[ix[indices]]


def run(config: M59Config) -> list[dict]:
    items = tuple(f'item-{i}' for i in range(config.candidates))
    pairs = list(permutations(items, 2))
    slates = list(permutations(items, config.k))
    n_pairs, n_slates = len(pairs), len(slates)
    pair_ids = {p: j for j, p in enumerate(pairs)}
    slate_ids = {s: j for j, s in enumerate(slates)}
    target_slate = np.array([next(slate_ids[s] for s in slates if s[:2] == p) for p in pairs])
    all_contexts = [p for p in pairs for _ in slates]
    all_slates = slates * n_pairs
    x = np.stack([features(s, items) for s in all_slates])
    ix = contextual_indices(all_slates, all_contexts, pairs)
    target_idx = np.arange(n_pairs) * n_slates + target_slate
    item_weights = np.linspace(-0.04, 0.04, config.candidates)
    item_ids = {item: j for j, item in enumerate(items)}
    base = np.array([0.3 + sum(item_weights[item_ids[item]] / (j + 1) for j, item in enumerate(s)) for s in all_slates])
    match = np.array([s[:2] == c for s, c in zip(all_slates, all_contexts)])
    truth = base + config.pair_bonus * match
    true_value = float(np.mean(truth[target_idx]))
    rows = []
    for n in config.requests:
        for exploration in config.explorations:
            p_target = (1 - exploration) + exploration / n_slates
            for trial in range(config.trials):
                rng = np.random.default_rng(config.seed + 100003 * n + 1009 * trial + int(exploration * 10000))
                contexts = rng.integers(n_pairs, size=n)
                logged = rng.integers(n_slates, size=n)
                exploit = rng.random(n) >= exploration
                logged[exploit] = target_slate[contexts[exploit]]
                selected = contexts * n_slates + logged
                observed = truth[selected] + rng.normal(0, config.noise, n)
                # Exact full-slate target/logging importance ratio.
                matches = logged == target_slate[contexts]
                weights = matches.astype(float) / p_target
                perm = rng.permutation(n)
                n_valid = max(2, int(n * config.validation_fraction))
                valid, evaluation = perm[:n_valid], perm[n_valid:]
                # Independent validation set is never used in reported OPE estimates.
                fold_indices = np.array_split(evaluation, config.folds)
                for name, contextual in [('additive', False), ('contextual', True)]:
                    q_logged = np.empty(len(evaluation))
                    q_target = np.empty(len(evaluation))
                    chosen_ridges = []
                    for fold in fold_indices:
                        training = np.setdiff1d(evaluation, fold, assume_unique=False)
                        # Select ridge only from independent validation outcomes.
                        losses = []
                        for ridge in config.ridges:
                            model = fit_model(x[selected], ix[selected], observed, training, ridge, n_pairs**2, contextual)
                            losses.append(float(np.mean((predict(model, x, ix, selected[valid]) - observed[valid])**2)))
                        chosen = config.ridges[int(np.argmin(losses))]
                        chosen_ridges.append(chosen)
                        model = fit_model(x[selected], ix[selected], observed, training, chosen, n_pairs**2, contextual)
                        # Use explicit row-to-position mapping: folds are not sorted.
                        locations = {int(row): j for j, row in enumerate(evaluation)}
                        dest = np.array([locations[int(row)] for row in fold])
                        q_logged[dest] = predict(model, x, ix, selected[fold])
                        q_target[dest] = predict(model, x, ix, target_idx[contexts[fold]])
                    w = weights[evaluation]
                    y = observed[evaluation]
                    dm = float(np.mean(q_target))
                    dr = float(np.mean(q_target + w * (y - q_logged)))
                    clipped = float(np.mean(q_target + np.minimum(w, config.clip) * (y - q_logged)))
                    estimates = {'dm': dm, 'dr': dr, 'clipped-dr': clipped}
                    if name == 'additive':
                        estimates['ips'] = float(np.mean(w * y))
                        estimates['snips'] = float(np.sum(w * y) / np.sum(w)) if np.sum(w) else float('nan')
                    ess = float(np.sum(w)**2 / np.sum(w**2)) if np.sum(w**2) else 0.0
                    for estimator, value in estimates.items():
                        rows.append(dict(requests=n, exploration=exploration, trial=trial,
                                         model=name if estimator not in ('ips', 'snips') else 'propensity',
                                         estimator=estimator, estimate=value, true_value=true_value,
                                         error=value - true_value, selected_ridge=float(np.mean(chosen_ridges)),
                                         eval_size=len(evaluation), target_matches=int(np.sum(matches[evaluation])),
                                         ess=ess, max_weight=float(np.max(w))))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['requests'], row['exploration'], row['model'], row['estimator']), []).append(row)
    result = []
    for (n, exploration, model, estimator), group in sorted(groups.items()):
        errors = np.array([r['error'] for r in group], dtype=float)
        finite = np.isfinite(errors)
        result.append(dict(requests=n, exploration=exploration, model=model, estimator=estimator,
                           trials=len(group), valid_trials=int(np.sum(finite)),
                           true_value=group[0]['true_value'],
                           bias=float(np.mean(errors[finite])) if np.any(finite) else float('nan'),
                           rmse=float(np.sqrt(np.mean(errors[finite]**2))) if np.any(finite) else float('nan'),
                           mean_ess=float(np.mean([r['ess'] for r in group])),
                           mean_selected_ridge=float(np.mean([r['selected_ridge'] for r in group]))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    parser.add_argument('--explorations', type=float, nargs='+', default=[0.2, 1.0])
    parser.add_argument('--ridges', type=float, nargs='+', default=[0.01, 1, 10])
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--pair-bonus', type=float, default=0.2)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m59_dr'))
    a = parser.parse_args(argv)
    config = M59Config(a.trials, tuple(a.requests), tuple(a.explorations), tuple(a.ridges),
                       a.candidates, a.k, a.pair_bonus, a.noise, a.folds,
                       a.validation_fraction, a.clip, a.seed)
    rows = run(config)
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summarize(rows))
    print(f'Wrote {len(rows)} rows to {a.output_dir}')


if __name__ == '__main__':
    main()
