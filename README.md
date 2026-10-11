# OPEBenchLab

A reproducible learning and benchmarking project for **off-policy evaluation (OPE)** in recommender systems.

## Research question

When can logged recommendation data reliably identify a better policy, and when do poor overlap, model misspecification, leakage, and ranking interactions make offline estimates misleading?

## Roadmap

- **M0 (this patch):** repository scaffold, data contracts, oracle isolation rules, and schema tests.
- **M1:** synthetic contextual-bandit generator, stochastic logging, and independent oracle policy values.
- **M2:** IPS, SNIPS, DM, DR, Switch-DR, and estimator tests.
- **M3:** Monte Carlo experiments, bias/variance/RMSE, confidence intervals, overlap, and ESS.
- **M4:** ML validity, point-in-time features, cross-fitting, and leakage stress tests.
- **M5:** top-K slate logging and evaluation under position and interaction effects.
- **M6:** real randomized logs, reporting, and policy-selection case studies.

## Setup

```bash
python -m pip install -e '.[dev]'
pytest
```

## Data contracts

The schema module defines typed records for a pre-decision request, a logged bandit event, and an **evaluation-only** oracle result. The oracle is deliberately kept separate from observed logs; later benchmark code must not feed it into an estimator or reward-model training.

The M0 schema uses single-action contextual bandits. Slates and richer candidate metadata are deferred to the slate milestone.

## M1: Synthetic contextual-bandit data

```python
from opebenchlab.synthetic import SyntheticConfig, make_dataset

data = make_dataset(SyntheticConfig(n_requests=1000, seed=42))
print(data.requests[0])
print(data.events[0])
```

`SyntheticWorld` holds the hidden click mechanism; `generate_requests` creates
pre-action features; `log_interactions` samples actions and clicks while recording
exact action propensities. `oracle_policy_value` computes the exact **conditional**
expected policy reward over the generated request set, not the population value.
Keep oracle outputs out of estimators and training. This milestone supports
single-item actions only; top-K slates arrive later.

## M6.0: Unified reproducibility wrapper

Use `python -m opebenchlab.m60_benchmark list` to see supported modern slate
benchmarks. Run one with `python -m opebenchlab.m60_benchmark run m512-support
--output-dir reports/m60_example --trials 1 --requests 500 --explorations 0.2
--ridges 0.1`. This leaves the original estimator code unchanged and writes a
provenance manifest alongside the original CSVs. See
[the M6.0 reproducibility guide](docs/m60_reproducibility.md) for the full
benchmark command, optional plotting, and interpretation caveats.

### M6.1: Offline reporting and uncertainty audit

After running `m60_benchmark run m512-support`, generate verified coverage and
paired-bootstrap summaries without rerunning the experiment:

```bash
python -m opebenchlab.m61_report --input-dir reports/m60_support \
  --output-dir reports/m61_support --bootstrap 2000 --seed 42
```

See [M6.1 reporting protocol](docs/m61_reporting.md) for interpretation and limitations.

### M6.2 staged robustness protocol

Run `python -m opebenchlab.m62_robustness --help` and see
[`docs/m62_robustness.md`](docs/m62_robustness.md) for the predefined replication,
support, and additive-negative-control suites. Interval calibration and a truly
misspecified interaction mechanism are explicitly deferred rather than inferred
from unsupported output fields.
