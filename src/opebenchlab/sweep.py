"""Reproducible exploration sweeps for the single-action OPE benchmark.

Run with: python -m opebenchlab.sweep --output-dir reports/m3_exploration
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from opebenchlab.benchmark import BenchmarkConfig, run_benchmark, summarize


@dataclass(frozen=True)
class SweepConfig:
    explorations: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)
    n_trials: int = 20
    n_requests: int = 1000
    n_bootstrap: int = 100
    seed: int = 42

    def __post_init__(self):
        if not self.explorations or len(set(self.explorations)) != len(self.explorations):
            raise ValueError("explorations must be nonempty and unique")
        for exploration in self.explorations:
            BenchmarkConfig(n_trials=self.n_trials, n_requests=self.n_requests,
                            n_bootstrap=self.n_bootstrap, seed=self.seed,
                            exploration=exploration)


def run_sweep(config: SweepConfig) -> tuple[dict, ...]:
    """Return one summary row per exploration level and estimator."""
    rows = []
    for exploration in config.explorations:
        benchmark = BenchmarkConfig(n_trials=config.n_trials,
                                    n_requests=config.n_requests,
                                    n_bootstrap=config.n_bootstrap,
                                    seed=config.seed, exploration=exploration)
        for result in summarize(run_benchmark(benchmark)):
            rows.append({"exploration": exploration, **asdict(result)})
    return tuple(rows)


def write_csv(rows: tuple[dict, ...], destination: Path) -> None:
    if not rows:
        raise ValueError("cannot export an empty sweep")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: tuple[dict, ...], output_dir: Path) -> None:
    """Create RMSE and ESS figures; matplotlib is an optional visualization extra."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib: python -m pip install matplotlib") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    estimators = list(dict.fromkeys(row["estimator"] for row in rows))
    for metric, filename, ylabel in (
        ("rmse", "rmse_vs_exploration.png", "RMSE vs conditional oracle value"),
        ("mean_ess", "ess_vs_exploration.png", "Mean effective sample size"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for estimator in estimators:
            series = sorted((r for r in rows if r["estimator"] == estimator),
                            key=lambda r: r["exploration"])
            ax.plot([r["exploration"] for r in series],
                    [r[metric] for r in series], marker="o", label=estimator)
        ax.set(xlabel="Logging exploration (epsilon)", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m3_exploration"))
    parser.add_argument("--explorations", type=float, nargs="+",
                        default=list(SweepConfig.explorations))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    config = SweepConfig(tuple(args.explorations), args.trials, args.requests,
                         args.bootstrap, args.seed)
    rows = run_sweep(config)
    write_csv(rows, args.output_dir / "summary.csv")
    if not args.no_plots:
        plot_sweep(rows, args.output_dir)
    print(f"Saved {len(rows)} summary rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
