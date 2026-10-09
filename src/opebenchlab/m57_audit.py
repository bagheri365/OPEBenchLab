"""M5.7: audit whether target-relative pair rewards live in fixed item-pair features.

This is a controlled feature-space diagnostic, not a deployable OPE estimator.
The injected bonus is known only to the audit's synthetic data generator.
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
class M57Config:
    trials: int = 10
    requests: tuple[int, ...] = (500, 2000)
    ridges: tuple[float, ...] = (0.001, 1.0, 10.0)
    candidates: int = 5
    k: int = 3
    pair_bonus: float = 0.2
    noise: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if self.trials < 1 or not self.requests or any(n < 4 for n in self.requests):
            raise ValueError('invalid trial/request count')
        if not 2 <= self.k <= self.candidates <= 7:
            raise ValueError('invalid slate dimensions')
        if not self.ridges or any(not np.isfinite(r) or r <= 0 for r in self.ridges):
            raise ValueError('invalid ridges')
        if not np.isfinite(self.pair_bonus) or self.pair_bonus < 0:
            raise ValueError('invalid pair bonus')
        if not np.isfinite(self.noise) or self.noise < 0:
            raise ValueError('invalid noise')


def audit_design(candidates: int, k: int):
    """Enumerate every slate and every possible target pair (target is contextual)."""
    items = tuple(f'item-{i}' for i in range(candidates))
    slates = list(permutations(items, k))
    # Each target pair can occur with every logged slate; contexts may vary.
    pairs = list(permutations(items, 2))
    observed = [s for _ in pairs for s in slates]
    targets = [p for p in pairs for _ in slates]
    x_add = np.stack([features(s, items) for s in observed])
    x_pair = np.stack([features(s, items, True) for s in observed])
    x_target = np.stack([targeted_features(s, p, items) for s, p in zip(observed, targets)])
    indicator = np.array([float(s[:2] == p) for s, p in zip(observed, targets)])
    return items, slates, pairs, x_add, x_pair, x_target, indicator


def projection_residual(x: np.ndarray, signal: np.ndarray) -> float:
    """RMSE of best possible linear representation, independent of ridge."""
    beta = np.linalg.lstsq(x, signal, rcond=None)[0]
    return float(np.sqrt(np.mean((x @ beta - signal) ** 2)))


def run(config: M57Config) -> list[dict]:
    items, slates, pairs, xa, xp, xt, signal = audit_design(config.candidates, config.k)
    matrices = {'additive': xa, 'generic-pairwise': xp, 'target-pair-diagnostic': xt}
    # Full factorial design is independent of training sampling and tests representability.
    residuals = {name: projection_residual(x, signal) for name, x in matrices.items()}
    ranks = {name: int(np.linalg.matrix_rank(x)) for name, x in matrices.items()}
    rng_count = len(signal)
    rows = []
    for n in config.requests:
        for trial in range(config.trials):
            rng = np.random.default_rng(config.seed + 100003 * n + trial)
            # Independent context/target pair and logged slate; aggregate reward only.
            train_idx = rng.integers(rng_count, size=n)
            eval_idx = rng.integers(rng_count, size=max(n, 1000))
            y_train = config.pair_bonus * signal[train_idx] + rng.normal(0, config.noise, n)
            truth = config.pair_bonus * signal[eval_idx]
            for ridge in config.ridges:
                for name, x in matrices.items():
                    beta = fit_ridge(x[train_idx], y_train, ridge)
                    predicted = x[eval_idx] @ beta
                    # Target-policy evaluation: first two items match target pair.
                    target_indices = eval_idx[signal[eval_idx] == 1]
                    if len(target_indices) == 0:
                        raise RuntimeError('insufficient target matches in evaluation draw')
                    target_pred = x[target_indices] @ beta
                    rows.append(dict(requests=n, trial=trial, ridge=ridge, model=name,
                                     slate_count=len(slates), target_pairs=len(pairs),
                                     exhaustive_rows=rng_count, design_columns=x.shape[1],
                                     design_rank=ranks[name], representation_rmse=residuals[name] * config.pair_bonus,
                                     train_pair_matches=int(signal[train_idx].sum()),
                                     logged_rmse=float(np.sqrt(np.mean((predicted - truth) ** 2))),
                                     target_value=float(np.mean(target_pred)),
                                     target_error=float(np.mean(target_pred) - config.pair_bonus),
                                     target_rmse=float(np.sqrt(np.mean((target_pred - config.pair_bonus) ** 2)))))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        groups.setdefault((r['requests'], r['ridge'], r['model']), []).append(r)
    result = []
    for (n, ridge, model), group in sorted(groups.items()):
        first = group[0]
        result.append(dict(requests=n, ridge=ridge, model=model, trials=len(group),
                           slate_count=first['slate_count'], target_pairs=first['target_pairs'],
                           design_columns=first['design_columns'], design_rank=first['design_rank'],
                           representation_rmse=first['representation_rmse'],
                           mean_train_pair_matches=float(np.mean([r['train_pair_matches'] for r in group])),
                           logged_rmse=float(np.mean([r['logged_rmse'] for r in group])),
                           target_bias=float(np.mean([r['target_error'] for r in group])),
                           target_rmse=float(np.sqrt(np.mean([r['target_error']**2 for r in group])))))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    p.add_argument('--ridges', type=float, nargs='+', default=[0.001, 1, 10])
    p.add_argument('--candidates', type=int, default=5)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--pair-bonus', type=float, default=0.2)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m57_audit'))
    a = p.parse_args(argv)
    cfg = M57Config(a.trials, tuple(a.requests), tuple(a.ridges), a.candidates,
                    a.k, a.pair_bonus, a.noise, a.seed)
    rows = run(cfg)
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summarize(rows))
    print(f'Wrote {len(rows)} rows to {a.output_dir}')


if __name__ == '__main__':
    main()
