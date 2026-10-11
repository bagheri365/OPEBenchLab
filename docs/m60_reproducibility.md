# M6.0: Reproducible benchmark entry point

M6.0 **wraps existing milestone CLIs without changing estimator implementations**.
The registry is intentionally limited to M5.9–M5.12, which expose comparable
command-line experiment interfaces. Other historical modules remain runnable
using their original `python -m opebenchlab.<module>` commands.

## Install and inspect

```bash
python -m pip install -e '.[dev]'
python -m opebenchlab.m60_benchmark list
pytest -q
```

## Reproduce a compact support-aware benchmark

```bash
python -m opebenchlab.m60_benchmark run m512-support \
  --output-dir reports/m60_support_example \
  --trials 30 --requests 500 2000 10000 \
  --explorations 0.05 0.2 0.5 1.0 \
  --ridges 0.001 0.01 0.1 1 10 \
  --seed 42
```

Each run writes the original milestone CSV files and `manifest.json` with the
invocation, Python version, UTC start, subprocess exit status, output columns,
row counts and SHA-256 digests. Use a **new or empty output directory** for each
run; the wrapper refuses to overwrite existing files.

The manifest records the command but does **not** freeze operating-system,
NumPy, or other dependency versions, nor does it prove deterministic results
across platforms. Record your Git commit and installed dependency versions
separately for strict reproduction.

## Optional figure

```bash
python -m pip install matplotlib
python -m opebenchlab.m60_benchmark plot \
  --summary reports/m60_support_example/summary.csv \
  --output reports/m60_support_example/conditional_rmse.png
```

The plot shows **conditional** RMSE, not unconditional error; check `coverage`
and `undefined_trials` in `summary.csv` before comparing methods. A finite
fallback estimate does not imply that its error is small.

## Interpretation boundaries

- Synthetic reward mechanisms can favor representations aligned with their construction.
- Full-slate SNIPS can be undefined with zero exact target-slate matches.
- M5.12 support-aware selection inspects evaluation **actions/propensities**, not evaluation rewards.
- Validation-based selection and evaluation-action-based fallback are different protocols.
- Paired squared-error comparisons exclude trials with undefined members and report the exclusions.
- This consolidation milestone does not establish real-world generalization or calibrated confidence intervals.
