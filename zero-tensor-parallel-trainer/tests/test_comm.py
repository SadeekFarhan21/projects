import pytest

from dtrain.dist_utils import launch
from workers import ring_vs_native, stats_formula


@pytest.mark.parametrize("world,numel", [(2, 1000), (3, 1001), (4, 7)])
def test_ring_all_reduce_matches_native(world, numel):
    diffs = launch(ring_vs_native, world, numel)
    assert max(diffs) < 1e-12


def test_wire_byte_accounting():
    world = 4
    s = launch(stats_formula, world)[0]
    nbytes = 1000 * 4
    assert s["wire_bytes"]["all_reduce"] == int(2 * (world - 1) / world * nbytes)
    assert s["wire_bytes"]["reduce_scatter"] == int((world - 1) / world * nbytes)
    assert s["wire_bytes"]["all_gather"] == int((world - 1) / world * nbytes)
    # reduce_scatter + all_gather == all_reduce: why ZeRO-1 costs no extra bandwidth
    assert (s["wire_bytes"]["reduce_scatter"] + s["wire_bytes"]["all_gather"]
            == s["wire_bytes"]["all_reduce"])
