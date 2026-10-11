"""M6.0: reproducible, non-invasive entry point for existing benchmarks.

Run legacy modules unchanged in a subprocess; store invocation and artifact
hashes so a benchmark can be audited without changing its statistical logic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

EXPERIMENTS = {
    'm59-dr': ('opebenchlab.m59_dr', ('trials.csv', 'summary.csv')),
    'm510-adaptive': ('opebenchlab.m510_adaptive', ('trials.csv', 'summary.csv')),
    'm511-validated': ('opebenchlab.m511_validated', ('trials.csv', 'summary.csv')),
    'm512-support': ('opebenchlab.m512_support', ('trials.csv', 'summary.csv', 'paired.csv')),
}
SCHEMA_VERSION = 'm6.0.1'


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_csv(path: Path) -> dict:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        count = sum(1 for _ in reader)
    return {'rows': count, 'columns': columns, 'sha256': file_digest(path)}


def execute(experiment: str, output_dir: Path, arguments: list[str]) -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment: {experiment}')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Output directory must be empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    module, expected = EXPERIMENTS[experiment]
    command = [sys.executable, '-m', module, *arguments, '--output-dir', str(output_dir.resolve())]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    artifacts = {name: inspect_csv(output_dir / name) for name in expected if (output_dir / name).is_file()}
    manifest = {
        'schema_version': SCHEMA_VERSION, 'experiment': experiment,
        'module': module, 'started_utc': started,
        'python_version': sys.version.split()[0],
        'argv': arguments, 'command': command,
        'returncode': result.returncode, 'artifacts': artifacts,
        'stdout': result.stdout, 'stderr': result.stderr,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Experiment failed (exit {result.returncode}); see {output_dir / "manifest.json"}')
    missing = set(expected) - artifacts.keys()
    if missing:
        raise RuntimeError(f'Missing expected artifacts: {sorted(missing)}')
    return manifest


def report(summary: Path, output: Path) -> None:
    """Create an RMSE chart for M5.12-style summary CSVs; never recalculate results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Install matplotlib to generate plots: pip install matplotlib') from exc
    with summary.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'requests', 'exploration', 'estimator', 'conditional_rmse', 'coverage'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Summary must contain {sorted(required)}')
    fig, ax = plt.subplots(figsize=(8, 5))
    for (eps, estimator) in sorted({(r['exploration'], r['estimator']) for r in rows}):
        subset = sorted((r for r in rows if r['exploration'] == eps and r['estimator'] == estimator), key=lambda r: int(r['requests']))
        xs, ys = [], []
        for r in subset:
            v = float(r['conditional_rmse'])
            if v >= 0 and v < float('inf'):
                xs.append(int(r['requests']))
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker='o', label=f'{estimator} (eps={eps})')
    ax.set(xlabel='Total requests', ylabel='Conditional RMSE', title='Slate OPE: conditional RMSE (see coverage in CSV)')
    ax.set_xscale('log')
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List supported benchmark entry points')
    run = sub.add_parser('run', help='Run an unchanged milestone CLI with provenance')
    run.add_argument('experiment', choices=sorted(EXPERIMENTS))
    run.add_argument('--output-dir', type=Path, required=True)
    plot = sub.add_parser('plot', help='Plot M5.12 conditional RMSE')
    plot.add_argument('--summary', type=Path, required=True)
    plot.add_argument('--output', type=Path, required=True)
    a, unknown = p.parse_known_args(argv)
    if a.command == 'list':
        if unknown:
            p.error(f'unrecognized arguments: {unknown}')
        for name, (module, outputs) in sorted(EXPERIMENTS.items()):
            print(f'{name}: {module} -> {", ".join(outputs)}')
    elif a.command == 'run':
        if '--output-dir' in unknown:
            p.error('Pass --output-dir only once, before experiment-specific flags')
        forwarded = unknown[1:] if unknown and unknown[0] == '--' else unknown
        manifest = execute(a.experiment, a.output_dir, forwarded)
        print(f'Wrote manifest and {len(manifest["artifacts"])} CSV artifacts to {a.output_dir}')
    else:
        if unknown:
            p.error(f'unrecognized arguments: {unknown}')
        report(a.summary, a.output)
        print(f'Wrote {a.output}')


if __name__ == '__main__':
    main()
