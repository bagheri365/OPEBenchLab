"""M6.3: diagnostic decomposition of contextual slate DR (no estimator changes).

DR error = target direct-model error + weighted residual correction.
The correction includes sampling noise; covariance is descriptive, not causal.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from opebenchlab.m511_validated import M511Config
from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m55_benchmark import features
from opebenchlab.m58_contextual import contextual_indices
from opebenchlab.m59_dr import fit_model, predict


@dataclass(frozen=True)
class Config:
    trials: int = 30
    requests: tuple[int, ...] = (500, 2000, 10000)
    explorations: tuple[float, ...] = (0.5, 1.0)
    seeds: tuple[int, ...] = (42, 1729)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    candidates: int = 5
    k: int = 3
    pair_bonus: float = 0.2
    noise: float = 0.05
    train_fraction: float = 0.4
    validation_fraction: float = 0.25

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be nonempty and unique')
        M511Config(trials=self.trials, requests=self.requests, explorations=self.explorations,
                   ridges=self.ridges, candidates=self.candidates, k=self.k,
                   pair_bonus=self.pair_bonus, noise=self.noise,
                   train_fraction=self.train_fraction, validation_fraction=self.validation_fraction)


def decompose(q_target, w, observed, q_logged, true_value):
    """Exact finite-sample identity, including zero-match samples."""
    dm = float(np.mean(q_target))
    correction = float(np.mean(w * (observed - q_logged)))
    dm_error = dm - true_value
    dr_error = dm_error + correction
    return dict(dm=dm, correction=correction, dr=dm+correction,
                dm_error=dm_error, dr_error=dr_error,
                dm_error_sq=dm_error**2, correction_sq=correction**2,
                cross_term=2*dm_error*correction, dr_error_sq=dr_error**2)


def run(config: Config):
    items = tuple(f'item-{i}' for i in range(config.candidates))
    pairs = list(permutations(items, 2))
    slates = list(permutations(items, config.k))
    n_pairs, n_slates = len(pairs), len(slates)
    slate_ids = {s: j for j, s in enumerate(slates)}
    target_slate = np.array([next(slate_ids[s] for s in slates if s[:2] == p) for p in pairs])
    contexts_all = [p for p in pairs for _ in slates]
    slates_all = slates * n_pairs
    x = np.stack([features(s, items) for s in slates_all])
    ix = contextual_indices(slates_all, contexts_all, pairs)
    target_idx = np.arange(n_pairs) * n_slates + target_slate
    item_weights = np.linspace(-0.04, 0.04, config.candidates)
    item_ids = {item: j for j, item in enumerate(items)}
    base = np.array([0.3 + sum(item_weights[item_ids[item]] / (j+1) for j, item in enumerate(s))
                     for s in slates_all])
    matches = np.array([s[:2] == c for s, c in zip(slates_all, contexts_all)])
    rows = []
    for mechanism, bonus in (('additive', 0.0), ('contextual-pairwise', config.pair_bonus)):
        truth = base + bonus * matches
        true_value = float(np.mean(truth[target_idx]))
        for seed in config.seeds:
            for n in config.requests:
                for exploration in config.explorations:
                    p_target = 1-exploration+exploration/n_slates
                    for trial in range(config.trials):
                        # Common random numbers across mechanisms: contexts, slates,
                        # noise and splits are identical for paired settings.
                        rng = np.random.default_rng(seed+100003*n+1009*trial+int(exploration*10000))
                        contexts = rng.integers(n_pairs, size=n)
                        logged = rng.integers(n_slates, size=n)
                        exploit = rng.random(n) >= exploration
                        logged[exploit] = target_slate[contexts[exploit]]
                        selected = contexts*n_slates+logged
                        observed = truth[selected] + rng.normal(0, config.noise, n)
                        weights = (logged == target_slate[contexts]).astype(float)/p_target
                        perm = rng.permutation(n)
                        n_train = int(n*config.train_fraction)
                        n_valid = int(n*config.validation_fraction)
                        train = perm[:n_train]
                        evaluation = perm[n_train+n_valid:]
                        subtrain = train[:max(2, int(0.8*len(train)))]
                        tune = train[len(subtrain):]
                        losses = []
                        for ridge in config.ridges:
                            model = fit_model(x[selected], ix[selected], observed,
                                              subtrain, ridge, n_pairs**2, True)
                            prediction = predict(model, x, ix, selected[tune])
                            losses.append(float(np.mean((prediction-observed[tune])**2)))
                        ridge = config.ridges[int(np.argmin(losses))]
                        model = fit_model(x[selected], ix[selected], observed,
                                          train, ridge, n_pairs**2, True)
                        q_logged = predict(model, x, ix, selected[evaluation])
                        q_target = predict(model, x, ix, target_idx[contexts[evaluation]])
                        w = weights[evaluation]
                        y = observed[evaluation]
                        parts = decompose(q_target, w, y, q_logged, true_value)
                        target_truth = truth[target_idx[contexts[evaluation]]]
                        support = int(np.count_nonzero(w))
                        sw2 = float(np.dot(w,w))
                        rows.append(dict(mechanism=mechanism, seed=seed, requests=n,
                                         exploration=exploration, trial=trial,
                                         candidates=config.candidates, k=config.k,
                                         selected_ridge=ridge, train_size=len(train),
                                         test_size=len(evaluation), true_value=true_value,
                                         target_matches=support,
                                         test_ess=float(w.sum()**2/sw2) if sw2 else 0.0,
                                         target_prediction_rmse=float(np.sqrt(np.mean((q_target-target_truth)**2))),
                                         logged_prediction_rmse=float(np.sqrt(np.mean((q_logged-truth[selected[evaluation]])**2))),
                                         snips=float(np.dot(w,y)/w.sum()) if support else float('nan'),
                                         **parts))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['mechanism'], row['requests'], row['exploration'])
        groups.setdefault(key, []).append(row)
    output = []
    for (mechanism, n, exploration), group in sorted(groups.items()):
        a = np.array([r['dm_error'] for r in group])
        b = np.array([r['correction'] for r in group])
        snips = np.array([r['snips']-r['true_value'] for r in group])
        valid = snips[np.isfinite(snips)]
        output.append(dict(mechanism=mechanism, requests=n, exploration=exploration,
                           trials=len(group), zero_match_trials=sum(r['target_matches']==0 for r in group),
                           mean_test_ess=float(np.mean([r['test_ess'] for r in group])),
                           dm_bias=float(np.mean(a)), correction_mean=float(np.mean(b)),
                           dr_bias=float(np.mean(a+b)),
                           dm_mse=float(np.mean(a*a)), correction_second_moment=float(np.mean(b*b)),
                           cross_second_moment=float(2*np.mean(a*b)),
                           dr_mse=float(np.mean((a+b)**2)),
                           dm_correction_covariance=float(np.cov(a,b,ddof=1)[0,1]) if len(group)>1 else float('nan'),
                           snips_coverage=len(valid)/len(group),
                           snips_conditional_rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else float('nan'),
                           target_prediction_rmse=float(np.mean([r['target_prediction_rmse'] for r in group])),
                           logged_prediction_rmse=float(np.mean([r['logged_prediction_rmse'] for r in group]))))
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=30)
    p.add_argument('--requests', type=int, nargs='+', default=[500,2000,10000])
    p.add_argument('--explorations', type=float, nargs='+', default=[0.5,1.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[42,1729])
    p.add_argument('--ridges', type=float, nargs='+', default=[0.01,1.0,10.0])
    p.add_argument('--candidates', type=int, default=5)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--pair-bonus', type=float, default=0.2)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m63_diagnostics'))
    args = p.parse_args(argv)
    cfg = Config(trials=args.trials, requests=tuple(args.requests),
                 explorations=tuple(args.explorations), seeds=tuple(args.seeds),
                 ridges=tuple(args.ridges), candidates=args.candidates, k=args.k,
                 pair_bonus=args.pair_bonus, noise=args.noise)
    rows = run(cfg)
    pth = args.output_dir
    write_csv(pth/'components.csv', rows)
    write_csv(pth/'summary.csv', summarize(rows))
    pth.mkdir(parents=True, exist_ok=True)
    (pth/'protocol.json').write_text(json.dumps(dict(config=asdict(cfg),
        decomposition='DR error = DM error + weighted residual correction',
        mse_identity='E[DR error^2] = E[DM error^2] + E[correction^2] + 2 E[DM error * correction]',
        covariance='descriptive covariance across Monte Carlo trials, not causal attribution',
        target_prediction_rmse='held-out target contexts, noiseless synthetic reward',
        logged_prediction_rmse='held-out logged actions, noiseless synthetic reward',
        seed_pairing='common contexts, logging actions, noise and splits across reward mechanisms'), indent=2)+'\n')
    print(f'Wrote {len(rows)} diagnostic trials to {pth}')


if __name__ == '__main__':
    main()
