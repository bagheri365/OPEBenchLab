import pytest

from opebenchlab.m62_robustness import Protocol, configurations, execute


def test_protocol_validation():
    with pytest.raises(ValueError, match='<= 7'):
        Protocol(candidates=10)
    with pytest.raises(ValueError, match='seeds'):
        Protocol(seeds=(1, 1))


def test_grid_fixed_and_distinct():
    p = Protocol(seeds=(3,), suite='support')
    grid = list(configurations(p))
    assert len(grid) == 4
    assert len({(x[2], x[3]) for x in grid}) == 4


def test_small_replication_and_coverage():
    p = Protocol(suite='replication', trials=1, seeds=(42,), requests=(40,),
                 explorations=(1.0,), ridges=(1.0,))
    rows, summary, intervals = execute(p)
    assert len(rows) == 5
    assert len(summary) == 5
    assert len(intervals) == 2
    assert all(0 <= r['coverage'] <= 1 for r in summary)
    assert all(r['intervals_computed'] == 0 for r in intervals)


def test_additive_negative_control():
    p = Protocol(suite='mechanism', seeds=(42,))
    assert [x[1] for x in configurations(p)] == ['additive', 'contextual-pairwise']
    assert [x[4] for x in configurations(p)] == [0.0, 0.2]
