import numpy as np
import pytest

from opebenchlab.m42_benchmark import (M42Config, cross_fit, fit, paired_differences,
                                       projection_matrix, run, summarize)
from opebenchlab.synthetic import SyntheticConfig, make_dataset


def test_paired_and_summary():
    rows = run(M42Config(trials=3, requests=(30,), explorations=(0.2,),
                              capacities=(0, 8), folds=3))
    assert len(rows) == 12
    assert len(summarize(rows)) == 4
    paired = paired_differences(rows)
    assert len(paired) == 2
    assert all(r['trials'] == 3 for r in paired)
    for trial in range(3):
        group = [r for r in rows if r['trial'] == trial]
        assert len({r['truth'] for r in group}) == 1
        ess = [r['ess'] for r in group]
        assert ess == pytest.approx([ess[0]] * len(ess))


def test_reproducible():
    config = M42Config(trials=2, requests=(20,), explorations=(0.2,), capacities=(8,), folds=2)
    a, b = run(config), run(config)
    assert [r['estimate'] for r in a] == pytest.approx([r['estimate'] for r in b])


def test_crossfit_does_not_train_on_heldout_reward():
    from opebenchlab.synthetic import LoggedDataset
    from dataclasses import replace
    data = make_dataset(SyntheticConfig(n_requests=20, seed=7))
    projection = projection_matrix(8, 42)
    original = cross_fit(data, projection, 0.1, folds=4, seed=42)
    changed = LoggedDataset(data.requests, (replace(data.events[0], reward=1-data.events[0].reward),)
                            + data.events[1:])
    updated = cross_fit(changed, projection, 0.1, folds=4, seed=42)
    req = data.requests[0]
    for item in req.candidate_item_ids:
        assert original(req, item) == pytest.approx(updated(req, item))


def test_capacity_and_validation():
    assert projection_matrix(0, 42) is None
    assert projection_matrix(9, 42).shape == (5, 9)
    with pytest.raises(ValueError):
        projection_matrix(-1, 42)
    with pytest.raises(ValueError):
        M42Config(trials=1)
    data = make_dataset(SyntheticConfig(n_requests=10))
    with pytest.raises(ValueError):
        fit(data, None, 0)
