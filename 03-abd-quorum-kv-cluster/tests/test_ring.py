from collections import Counter

from kvstore.ring import HashRing


def test_preference_list_distinct_and_deterministic():
    ring = HashRing([f"n{i}" for i in range(5)], vnodes=64)
    for k in range(500):
        pl = ring.preference_list(f"key{k}", 3)
        assert len(pl) == 3 and len(set(pl)) == 3
        assert pl == HashRing([f"n{i}" for i in range(5)], vnodes=64).preference_list(f"key{k}", 3)


def test_n_capped_at_cluster_size():
    ring = HashRing(["a", "b"], vnodes=8)
    assert sorted(ring.preference_list("x", 3)) == ["a", "b"]


def test_load_is_roughly_balanced_with_vnodes():
    ring = HashRing([f"n{i}" for i in range(5)], vnodes=128)
    primaries = Counter(ring.preference_list(f"key{k}", 1)[0] for k in range(20000))
    shares = [c / 20000 for c in primaries.values()]
    assert len(shares) == 5
    assert max(shares) < 0.30 and min(shares) > 0.12  # ideal is 0.20


def test_adding_a_node_moves_about_one_over_n_keys():
    before = HashRing([f"n{i}" for i in range(4)], vnodes=128)
    after = HashRing([f"n{i}" for i in range(5)], vnodes=128)
    keys = [f"key{k}" for k in range(10000)]
    moved = sum(before.preference_list(k, 1) != after.preference_list(k, 1) for k in keys)
    # Ideal is 1/5 of keys; modulo hashing would move ~4/5.
    assert 0.12 < moved / len(keys) < 0.28
    # And every moved key moved to the new node.
    assert all(after.preference_list(k, 1) == ["n4"] for k in keys
               if before.preference_list(k, 1) != after.preference_list(k, 1))


def test_remove_node():
    ring = HashRing(["a", "b", "c"], vnodes=16)
    ring.remove_node("b")
    assert ring.nodes == ["a", "c"]
    assert all("b" not in ring.preference_list(f"k{i}", 2) for i in range(100))
