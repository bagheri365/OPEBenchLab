from opebenchlab.synthetic import (
    SyntheticConfig,
    make_dataset,
    uniform_probabilities,
)
from opebenchlab.reward_models import (
    fit_reward_model,
    cross_fit_reward_model,
    brier_score,
)
from opebenchlab.estimators import doubly_robust


def main():
    # Generate reproducible synthetic recommendation logs
    data = make_dataset(
        SyntheticConfig(
            n_requests=1000,
            seed=42,
        )
    )

    # Train reward model using the same evaluation data
    in_sample = fit_reward_model(data)

    # Generate out-of-fold reward predictions
    cross_fitted = cross_fit_reward_model(
        data,
        folds=5,
        seed=42,
    )

    # Evaluate both with Doubly Robust
    print("\nIn-sample vs Cross-fitted DR")
    print("-" * 65)

    for name, model in [
        ("In-sample", in_sample),
        ("Cross-fitted", cross_fitted),
    ]:
        score = brier_score(data, model)

        result = doubly_robust(
            data,
            uniform_probabilities,
            model,
        )

        print(
            f"{name:15s} | "
            f"Brier: {score:.5f} | "
            f"DR value: {result.value:.5f} | "
            f"ESS: {result.effective_sample_size:.1f}"
        )


if __name__ == "__main__":
    main()
