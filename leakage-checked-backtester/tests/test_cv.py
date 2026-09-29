import numpy as np
import pytest

from qrp.cv import check_split, purged_splits


@pytest.mark.parametrize("kind", ["walk_forward", "kfold"])
@pytest.mark.parametrize("h,emb", [(1, 0), (5, 0), (5, 10), (21, 5)])
def test_purged_splits_are_clean(kind, h, emb):
    splits = purged_splits(1000, 6, label_horizon=h, embargo=emb, kind=kind, min_train=200)
    assert splits
    for sp in splits:
        assert check_split(sp, h, emb) == []


def test_walk_forward_never_trains_on_future():
    for sp in purged_splits(800, 5, label_horizon=5, embargo=3, min_train=100):
        assert sp.train.max() < sp.test.min()
        # exactly label_horizon rows are purged before the test block
        assert sp.train.max() == sp.test.min() - 5 - 1


def test_test_blocks_tile_the_out_of_sample_period():
    splits = purged_splits(1000, 4, label_horizon=5, min_train=200)
    tests = np.concatenate([s.test for s in splits])
    np.testing.assert_array_equal(tests, np.arange(200, 1000))


def test_rolling_train_window():
    for sp in purged_splits(1000, 4, label_horizon=5, min_train=300, train_window=250):
        assert len(sp.train) <= 250


def test_unpurged_split_is_detected():
    """With purge off the checker must flag the overlap (this is the leak purging removes)."""
    splits = purged_splits(500, 4, label_horizon=10, kind="kfold", purge=False)
    errs = [e for sp in splits for e in check_split(sp, 10, 0)]
    assert any("not purged" in e for e in errs)


def test_kfold_embargo_gap():
    sp = purged_splits(1000, 5, label_horizon=5, embargo=20, kind="kfold")[1]
    a, b = sp.test_range
    after = sp.train[sp.train > b]
    assert after.min() == b + 5 + 20 + 1
    before = sp.train[sp.train < a]
    assert before.max() == a - 5 - 1
