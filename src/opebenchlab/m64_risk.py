"""M6.4: observable target-distribution risk diagnostics, oracle-separated evaluation."""
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
                        # Observable diagnostics: training/validation rewards only; evaluation
                        # actions/propensities allowed for support, never evaluation rewards.
                        validation = perm[n_train:n_train+n_valid]
                        validation_pred = predict(model, x, ix, selected[validation])
                        validation_rmse = float(np.sqrt(np.mean((validation_pred-observed[validation])**2)))
                        training_pred = predict(model, x, ix, selected[train])
                        training_rmse = float(np.sqrt(np.mean((training_pred-observed[train])**2)))
                        ridge_target_means = []
                        for candidate_ridge in config.ridges:
                            alternative = fit_model(x[selected], ix[selected], observed,
                                                    train, candidate_ridge, n_pairs**2, True)
                            ridge_target_means.append(float(np.mean(predict(
                                alternative, x, ix, target_idx[contexts[evaluation]]))))
                        # Split-model disagreement is based on independent halves of training.
                        halves = np.array_split(train, 2)
                        half_target_means = []
                        for half in halves:
                            half_model = fit_model(x[selected], ix[selected], observed,
                                                   half, ridge, n_pairs**2, True)
                            half_target_means.append(float(np.mean(predict(
                                half_model, x, ix, target_idx[contexts[evaluation]]))))
                        observable = dict(
                            validation_rmse=validation_rmse,
                            training_rmse=training_rmse,
                            validation_gap=validation_rmse-training_rmse,
                            ridge_target_range=float(np.ptp(ridge_target_means)),
                            split_target_disagreement=abs(half_target_means[0]-half_target_means[1]),
                            validation_target_matches=int(np.count_nonzero(weights[validation])),
                        )
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
                                         **observable, **parts))
    return rows


def summarize(rows):
    """Within-configuration top-quartile warning precision and false-alarm rates.

    Risk is oracle-labelled as DR absolute error >= the configuration's 75th
    percentile; thresholds are evaluation-only, never part of the warnings.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row['mechanism'], row['requests'], row['exploration']), []).append(row)
    out = []
    for (mechanism, n, exploration), group in sorted(groups.items()):
        errors = np.array([abs(r['dr_error']) for r in group])
        high_error = errors >= np.quantile(errors, .75)
        for signal in ('validation_rmse', 'validation_gap', 'ridge_target_range',
                       'split_target_disagreement', 'test_ess'):
            values = np.array([r[signal] for r in group], dtype=float)
            if signal == 'test_ess':
                warnings = values <= np.quantile(values, .25)
            else:
                warnings = values >= np.quantile(values, .75)
            out.append(dict(mechanism=mechanism, requests=n, exploration=exploration,
                            signal=signal, trials=len(group), high_error_count=int(high_error.sum()),
                            flagged=int(warnings.sum()), true_positives=int((warnings & high_error).sum()),
                            false_positives=int((warnings & ~high_error).sum()),
                            precision=float(np.mean(high_error[warnings])) if warnings.any() else float('nan'),
                            recall=float(np.mean(warnings[high_error])) if high_error.any() else float('nan'),
                            false_positive_rate=float(np.mean(warnings[~high_error])) if (~high_error).any() else float('nan'),
                            mean_dr_abs_error=float(np.mean(errors))))
    return out

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
    p.add_argument('--output-dir', type=Path, default=Path('reports/m64_risk'))
    args = p.parse_args(argv)
    cfg = Config(trials=args.trials, requests=tuple(args.requests),
                 explorations=tuple(args.explorations), seeds=tuple(args.seeds),
                 ridges=tuple(args.ridges), candidates=args.candidates, k=args.k,
                 pair_bonus=args.pair_bonus, noise=args.noise)
    rows = run(cfg)
    pth = args.output_dir
    write_csv(pth/'diagnostics.csv', rows)
    write_csv(pth/'risk_summary.csv', summarize(rows))
    pth.mkdir(parents=True, exist_ok=True)
    (pth/'protocol.json').write_text(json.dumps(dict(
        config=asdict(cfg),
        observable_signals=['validation_rmse', 'training_rmse', 'validation_gap',
                            'ridge_target_range', 'split_target_disagreement',
                            'validation_target_matches', 'target_matches', 'test_ess'],
        oracle_only=['true_value', 'dm_error', 'dr_error', 'target_prediction_rmse',
                     'logged_prediction_rmse', 'high_error_quartile_label'],
        evaluation_data_policy='Evaluation actions and propensities may enter support and target-prediction diagnostics; evaluation rewards are never used in warnings.',
        warning_rule='Top quartile per configuration for positive-risk signals; bottom quartile for ESS; ties flagged.',
        oracle_label='Top quartile absolute DR error within each configuration; retrospective evaluation only.',
        caution='Quantile thresholds are retrospective within-configuration diagnostics, not deployable calibrated alert cutoffs.',
        seed_pairing='Common contexts, logging actions, noise and splits across mechanisms'
    ), indent=2)+'\n')
    print(f'Wrote {len(rows)} diagnostic trials to {pth}')


if __name__ == '__main__':
    main()
