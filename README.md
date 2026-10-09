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
