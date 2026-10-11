"""M6.1 offline audit and paired-bootstrap reporting for M5.12 artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    artifacts = manifest.get('artifacts', {})
    if not artifacts:
        raise ValueError('Manifest contains no artifacts')
    for name, metadata in artifacts.items():
        if Path(name).name != name:
            raise ValueError(f'Unsafe artifact name: {name}')
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != metadata['columns'] or sum(1 for _ in reader) != metadata['rows']:
                raise ValueError(f'CSV schema or row count mismatch: {name}')
    return manifest


def analyze(trials: list[dict], *, bootstrap: int = 2000, seed: int = 42) -> tuple[list[dict], list[dict]]:
    if bootstrap < 100:
        raise ValueError('bootstrap must be at least 100')
    required = {'requests', 'exploration', 'trial', 'estimator', 'error'}
    if not trials or not required.issubset(trials[0]):
        raise ValueError(f'Trials CSV must contain {sorted(required)}')
    grouped: dict[tuple[str, str], dict[str, dict[str, dict]]] = {}
    for row in trials:
        key = (row['requests'], row['exploration'])
        by_trial = grouped.setdefault(key, {})
        estimators = by_trial.setdefault(row['trial'], {})
        if row['estimator'] in estimators:
            raise ValueError(f'Duplicate trial/estimator: {key}, {row["trial"]}, {row["estimator"]}')
        estimators[row['estimator']] = row
    coverage, paired = [], []
    rng = np.random.default_rng(seed)
    for (n, eps), trials_by_id in sorted(grouped.items(), key=lambda x: (int(x[0][0]), float(x[0][1]))):
        ids = sorted(trials_by_id, key=lambda x: int(x))
        names = sorted({name for t in trials_by_id.values() for name in t})
        for name in names:
            errors = [float(trials_by_id[t][name]['error']) if name in trials_by_id[t] else math.nan for t in ids]
            valid = np.array([e for e in errors if math.isfinite(e)])
            coverage.append(dict(requests=n, exploration=eps, estimator=name,
                                 trials=len(ids), finite_trials=len(valid), undefined_trials=len(ids)-len(valid),
                                 coverage=len(valid)/len(ids),
                                 conditional_rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else math.nan))
        differences = []
        excluded = 0
        for t in ids:
            g = trials_by_id[t]
            if 'snips' not in g or 'contextual-dr' not in g:
                excluded += 1
                continue
            a, b = float(g['snips']['error']), float(g['contextual-dr']['error'])
            if not (math.isfinite(a) and math.isfinite(b)):
                excluded += 1
            else:
                differences.append(b*b-a*a)
        diff = np.asarray(differences, dtype=float)
        if len(diff):
            draws = rng.choice(diff, size=(bootstrap, len(diff)), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            mean = float(diff.mean())
            wins = float(np.mean(diff < 0))
        else:
            lo = hi = mean = wins = math.nan
        paired.append(dict(requests=n, exploration=eps, trials=len(ids), paired_trials=len(diff),
                           excluded_nonfinite_or_missing=excluded,
                           mean_dr_minus_snips_squared_error=mean,
                           bootstrap_ci_lower=float(lo), bootstrap_ci_upper=float(hi),
                           dr_win_fraction=wins))
    return coverage, paired


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('No rows to write')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error('Output directory must differ from input directory')
    manifest = verify_manifest(args.input_dir)
    if manifest.get('experiment') != 'm512-support':
        parser.error('This reporting protocol currently requires an m512-support manifest')
    coverage, paired = analyze(read_csv(args.input_dir / 'trials.csv'), bootstrap=args.bootstrap, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'coverage.csv', coverage)
    write_csv(args.output_dir / 'paired_bootstrap.csv', paired)
    report = {'source_experiment': manifest['experiment'], 'source_manifest_sha256': hashlib.sha256(
        (args.input_dir / 'manifest.json').read_bytes()).hexdigest(),
        'bootstrap_resamples': args.bootstrap, 'bootstrap_seed': args.seed,
        'comparison': 'contextual-dr squared error minus snips squared error',
        'interval': '95% percentile bootstrap, paired over finite trials; exploratory, not simultaneous',
        'coverage_definition': 'finite estimates / all unique trial IDs in each configuration',
        'groups': len(paired)}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Verified manifest; wrote {len(coverage)} coverage rows and {len(paired)} paired comparisons')


if __name__ == '__main__':
    main()
