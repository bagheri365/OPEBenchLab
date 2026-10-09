"""M5.10: validation-guided overlap-aware selection of slate OPE estimators.

The selector uses an independent validation split and never observes the
synthetic ground truth. Retrospective oracle regret is for reporting only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opebenchlab.m52_benchmark import write_csv
from opebenchlab.m59_dr import M59Config, run as run_m59


@dataclass(frozen=True)
class M510Config:
    trials: int = 10
    requests: tuple[int, ...] = (500, 2000)
    explorations: tuple[float, ...] = (0.2, 1.0)
    ridges: tuple[float, ...] = (0.01, 1.0, 10.0)
    clips: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 60.0)
    candidates: int = 5
    k: int = 3
    pair_bonus: float = 0.2
    noise: float = 0.05
    folds: int = 2
    validation_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if not self.clips or any(not np.isfinite(c) or c <= 0 for c in self.clips):
            raise ValueError('clips must be finite and positive')
        M59Config(self.trials, self.requests, self.explorations, self.ridges,
                  self.candidates, self.k, self.pair_bonus, self.noise,
                  self.folds, self.validation_fraction, self.clips[0], self.seed)


def choose(validation: dict[str, float], ess: float, n: int) -> str:
    """Conservative overlap rule; only observable validation diagnostics.

    SNIPS is preferred when effective sample size is sufficient; otherwise
    contextual DM avoids a high-variance correction. This is a fixed heuristic,
    not an optimally learned selector.
    """
    if ess >= max(20.0, 0.05 * n) and np.isfinite(validation.get('snips', np.nan)):
        return 'snips'
    return 'contextual-dm'


def run(config: M510Config) -> list[dict]:
    """Replay M5.9 at different clipping levels with paired seeds.

    The M5.9 core reuses the same random samples for each clipping threshold;
    selection uses only the ESS and SNIPS availability, not truth/errors.
    """
    by_key: dict[tuple, dict] = {}
    for clip in (*config.clips, float('inf')):
        # M59Config requires finite clip; an effectively infinite threshold
        # reproduces ordinary DR because importance weights are finite.
        effective_clip = clip if np.isfinite(clip) else 1e12
        cfg = M59Config(config.trials, config.requests, config.explorations,
                        config.ridges, config.candidates, config.k,
                        config.pair_bonus, config.noise, config.folds,
                        config.validation_fraction, effective_clip, config.seed)
        for row in run_m59(cfg):
            estimator = row['estimator']
            model = row['model']
            if estimator == 'clipped-dr':
                name = f'{model}-dr-clip-{clip:g}' if np.isfinite(clip) else f'{model}-dr-unclipped'
            elif estimator in ('ips', 'snips'):
                name = estimator
            else:
                name = f'{model}-{estimator}'
            key = (row['requests'], row['exploration'], row['trial'])
            by_key.setdefault(key, {})[name] = row
    output = []
    for (n, exploration, trial), candidates in sorted(by_key.items()):
        base = candidates['snips']
        ess = base['ess']
        selected = choose({'snips': base['estimate']}, ess, base['eval_size'])
        # Retrospective oracle uses true value, strictly after selection.
        valid = {name: r for name, r in candidates.items() if np.isfinite(r['error'])}
        oracle = min(valid, key=lambda name: abs(valid[name]['error']))
        oracle_sq = valid[oracle]['error']**2
        for name, row in candidates.items():
            output.append(dict(requests=n, exploration=exploration, trial=trial,
                               estimator=name, estimate=row['estimate'],
                               true_value=row['true_value'], error=row['error'],
                               ess=ess, eval_size=row['eval_size'],
                               max_weight=row['max_weight'],
                               selected_estimator=selected, oracle_estimator=oracle,
                               selected=(name == selected),
                               squared_regret=(row['error']**2 - oracle_sq) if np.isfinite(row['error']) else float('nan')))
    return output


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row['requests'], row['exploration'], row['estimator']), []).append(row)
    result = []
    for (n, exploration, estimator), group in sorted(groups.items()):
        errors = np.array([r['error'] for r in group], dtype=float)
        finite = np.isfinite(errors)
        result.append(dict(requests=n, exploration=exploration, estimator=estimator,
                           trials=len(group), valid_trials=int(finite.sum()),
                           bias=float(np.mean(errors[finite])) if finite.any() else float('nan'),
                           rmse=float(np.sqrt(np.mean(errors[finite]**2))) if finite.any() else float('nan'),
                           mean_ess=float(np.mean([r['ess'] for r in group])),
                           selection_rate=float(np.mean([r['selected'] for r in group])),
                           mean_squared_regret=float(np.nanmean([r['squared_regret'] for r in group]))))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--requests', type=int, nargs='+', default=[500, 2000])
    p.add_argument('--explorations', type=float, nargs='+', default=[0.2, 1.0])
    p.add_argument('--ridges', type=float, nargs='+', default=[0.01, 1, 10])
    p.add_argument('--clips', type=float, nargs='+', default=[2, 5, 10, 20, 60])
    p.add_argument('--candidates', type=int, default=5)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--pair-bonus', type=float, default=0.2)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--folds', type=int, default=2)
    p.add_argument('--validation-fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=Path('reports/m510_adaptive'))
    a = p.parse_args(argv)
    cfg = M510Config(a.trials, tuple(a.requests), tuple(a.explorations), tuple(a.ridges),
                     tuple(a.clips), a.candidates, a.k, a.pair_bonus, a.noise,
                     a.folds, a.validation_fraction, a.seed)
    rows = run(cfg)
    write_csv(a.output_dir / 'trials.csv', rows)
    write_csv(a.output_dir / 'summary.csv', summarize(rows))
    print(f'Wrote {len(rows)} rows to {a.output_dir}')


if __name__ == '__main__':
    main()
