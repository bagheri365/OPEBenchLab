"""M5.11: conservative, validation-only selection between SNIPS and contextual DR.

The validation criterion is a heuristic based on estimated influence variance and
estimator agreement, not a guaranteed estimator of relative mean-squared error.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m55_benchmark import features
from opebenchlab.m58_contextual import contextual_indices
from opebenchlab.m59_dr import fit_model, predict


@dataclass(frozen=True)
class M511Config:
    trials: int = 10
    requests: tuple[int, ...] = (500, 2000)
    explorations: tuple[float, ...] = (0.2, 1.0)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    candidates: int = 5
    k: int = 3
    pair_bonus: float = 0.2
    noise: float = 0.05
    validation_fraction: float = 0.25
    train_fraction: float = 0.4
    variance_ratio: float = 0.8
    agreement_z: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or min(self.requests) < 30:
            raise ValueError('invalid trials or requests')
        if not 2 <= self.k <= self.candidates <= 7:
            raise ValueError('invalid slate dimensions')
        if not self.explorations or any(not 0 < e <= 1 for e in self.explorations):
            raise ValueError('invalid exploration')
        if not self.ridges or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('invalid ridge')
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1 - self.train_fraction:
            raise ValueError('invalid split fractions')
        if not 0 < self.variance_ratio < 1 or self.agreement_z <= 0:
            raise ValueError('invalid selection thresholds')
        if not 0 <= self.pair_bonus <= 1 or self.noise < 0:
            raise ValueError('invalid reward parameters')


def estimate(y: np.ndarray, w: np.ndarray, q_logged: np.ndarray, q_target: np.ndarray):
    """Estimates and sample-based standard errors from disjoint held-out rows."""
    n = len(y)
    sw = float(w.sum())
    snips = float(np.dot(w, y) / sw) if sw > 0 else float('nan')
    dr_contrib = q_target + w * (y - q_logged)
    dr = float(np.mean(dr_contrib))
    dr_se = float(np.std(dr_contrib, ddof=1) / np.sqrt(n)) if n > 1 else float('inf')
    snips_influence = w * (y - snips) / np.mean(w) if sw > 0 else np.full(n, np.nan)
    snips_se = float(np.std(snips_influence, ddof=1) / np.sqrt(n)) if n > 1 else float('inf')
    return dict(snips=snips, contextual_dr=dr, contextual_dm=float(np.mean(q_target)),
                snips_se=snips_se, contextual_dr_se=dr_se,
                ess=float(sw**2 / np.dot(w, w)) if np.dot(w, w) else 0.0)


def choose(stats: dict, variance_ratio: float = 0.8, agreement_z: float = 1.0) -> str:
    """Switch only on validation variance advantage and approximate agreement.

    This is an empirical heuristic; correlated influence functions and bias can
    make the agreement criterion conservative or misleading.
    """
    a, b = stats['snips_se'], stats['contextual_dr_se']
    if (all(np.isfinite(v) for v in (a, b, stats['snips'], stats['contextual_dr']))
            and b < variance_ratio * a
            and abs(stats['contextual_dr'] - stats['snips']) <= agreement_z * np.hypot(a, b)):
        return 'contextual-dr'
    return 'snips'


def run(config: M511Config) -> list[dict]:
    items = tuple(f'item-{i}' for i in range(config.candidates))
    pairs = list(permutations(items, 2))
    slates = list(permutations(items, config.k))
    n_pairs, n_slates = len(pairs), len(slates)
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
    output = []
    for n in config.requests:
        for exploration in config.explorations:
            p_target = 1 - exploration + exploration / n_slates
            for trial in range(config.trials):
                rng = np.random.default_rng(config.seed + 100003 * n + 1009 * trial + int(exploration * 10000))
                contexts = rng.integers(n_pairs, size=n)
                logged = rng.integers(n_slates, size=n)
                exploit = rng.random(n) >= exploration
                logged[exploit] = target_slate[contexts[exploit]]
                selected = contexts * n_slates + logged
                observed = truth[selected] + rng.normal(0, config.noise, n)
                weights = (logged == target_slate[contexts]).astype(float) / p_target
                perm = rng.permutation(n)
                n_train = int(n * config.train_fraction)
                n_valid = int(n * config.validation_fraction)
                train, valid, evaluation = perm[:n_train], perm[n_train:n_train+n_valid], perm[n_train+n_valid:]
                # Ridge tuning uses training-only internal holdout, never selection validation.
                subtrain = train[:max(2, int(0.8 * len(train)))]
                tune = train[len(subtrain):]
                losses = []
                for ridge in config.ridges:
                    candidate = fit_model(x[selected], ix[selected], observed, subtrain, ridge, n_pairs**2, True)
                    pred = predict(candidate, x, ix, selected[tune])
                    losses.append(float(np.mean((pred - observed[tune])**2)))
                ridge = config.ridges[int(np.argmin(losses))]
                model = fit_model(x[selected], ix[selected], observed, train, ridge, n_pairs**2, True)
                def heldout(indices):
                    return estimate(observed[indices], weights[indices],
                                    predict(model, x, ix, selected[indices]),
                                    predict(model, x, ix, target_idx[contexts[indices]]))
                validation = heldout(valid)
                selected_name = choose(validation, config.variance_ratio, config.agreement_z)
                test = heldout(evaluation)
                oracle = min(('snips', 'contextual-dr', 'contextual-dm'),
                             key=lambda name: abs(test[name.replace('-', '_')] - true_value))
                oracle_error_sq = (test[oracle.replace('-', '_')] - true_value)**2
                for name in ('snips', 'contextual-dr', 'contextual-dm', 'selected'):
                    actual = selected_name if name == 'selected' else name
                    value = test[actual.replace('-', '_')]
                    output.append(dict(requests=n, exploration=exploration, trial=trial,
                                       estimator=name, selected_estimator=selected_name,
                                       oracle_estimator=oracle, estimate=value,
                                       true_value=true_value, error=value-true_value,
                                       squared_regret=(value-true_value)**2-oracle_error_sq,
                                       validation_ess=validation['ess'], test_ess=test['ess'],
                                       validation_snips_se=validation['snips_se'],
                                       validation_dr_se=validation['contextual_dr_se'],
                                       validation_disagreement=validation['contextual_dr']-validation['snips'],
                                       selected_ridge=ridge, train_size=len(train),
                                       validation_size=len(valid), test_size=len(evaluation)))
    return output


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row['requests'], row['exploration'], row['estimator']), []).append(row)
    result = []
    for (n, exploration, estimator), group in sorted(groups.items()):
        errors = np.array([r['error'] for r in group])
        result.append(dict(requests=n, exploration=exploration, estimator=estimator,
                           trials=len(group), bias=float(np.mean(errors)),
                           rmse=float(np.sqrt(np.mean(errors**2))),
                           mean_squared_regret=float(np.mean([r['squared_regret'] for r in group])),
                           selected_dr_fraction=float(np.mean([r['selected_estimator'] == 'contextual-dr' for r in group])),
                           mean_validation_ess=float(np.mean([r['validation_ess'] for r in group]))))
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
    parser.add_argument('--validation-fraction', type=float, default=0.25)
    parser.add_argument('--train-fraction', type=float, default=0.4)
    parser.add_argument('--variance-ratio', type=float, default=0.8)
    parser.add_argument('--agreement-z', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m511_validated'))
    a = parser.parse_args(argv)
    cfg = M511Config(a.trials, tuple(a.requests), tuple(a.explorations), tuple(a.ridges),
                     a.candidates, a.k, a.pair_bonus, a.noise, a.validation_fraction,
                     a.train_fraction, a.variance_ratio, a.agreement_z, a.seed)
    rows = run(cfg)
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summarize(rows))
    print(f'Wrote {len(rows)} rows to {a.output_dir}')


if __name__ == '__main__':
    main()
