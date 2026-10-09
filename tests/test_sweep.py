import csv

import pytest

from opebenchlab.sweep import SweepConfig, main, run_sweep, write_csv


def test_sweep_rows_and_reproducibility():
    config = SweepConfig((0.2, 1.0), n_trials=2, n_requests=40,
                         n_bootstrap=3, seed=9)
    first = run_sweep(config)
    second = run_sweep(config)
    assert len(first) == 10
    assert [(r["exploration"], r["estimator"]) for r in first] == [
        (r["exploration"], r["estimator"]) for r in second]
    for a, b in zip(first, second):
        for key in ("bias", "variance", "rmse", "mean_ess", "mean_max_weight", "ci_coverage"):
            assert a[key] == pytest.approx(b[key], abs=1e-12, nan_ok=True)


def test_invalid_explorations():
    for levels in ((), (0.0,), (1.1,), (0.2, 0.2)):
        with pytest.raises(ValueError):
            SweepConfig(explorations=levels)


def test_csv_and_cli(tmp_path):
    destination = tmp_path / "results" / "summary.csv"
    assert main(["--output-dir", str(destination.parent), "--explorations", "0.2",
                 "--trials", "2", "--requests", "30", "--bootstrap", "3",
                 "--no-plots"]) == 0
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert set(rows[0]) >= {"exploration", "estimator", "rmse", "mean_ess"}
    with pytest.raises(ValueError):
        write_csv((), destination)


def test_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from opebenchlab.sweep import plot_sweep
    rows = run_sweep(SweepConfig((0.2,), n_trials=2, n_requests=30,
                                 n_bootstrap=3))
    plot_sweep(rows, tmp_path)
    assert (tmp_path / "rmse_vs_exploration.png").is_file()
    assert (tmp_path / "ess_vs_exploration.png").is_file()
