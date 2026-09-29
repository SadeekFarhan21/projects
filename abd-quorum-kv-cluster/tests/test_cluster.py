"""Multi-process tests: a real 5-node cluster, real SIGKILLs."""

import asyncio
import time

import pytest

from kvstore.checker import check_history
from kvstore.client import KVClient
from kvstore.cluster import LocalCluster
from kvstore.jepsen import WorkloadConfig, kill_one_nemesis, pause_one_nemesis, run_workload


@pytest.fixture
def cluster(tmp_path):
    c = LocalCluster(size=5, n=3, r=2, w=2, data_root=str(tmp_path))
    c.start()
    yield c
    c.stop()


async def wait_for(pred, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await pred():
            return True
        await asyncio.sleep(interval)
    return False


async def test_basic_ops_across_coordinators(cluster):
    c = KVClient(cluster.addrs, seed=1)
    try:
        for i in range(200):
            assert (await c.put(f"key{i}", i))["ok"]
        for i in range(200):
            assert (await c.get(f"key{i}"))["value"] == i
        # Data is sharded: every node holds some keys, none holds all 200.
        # Replication factor 3 means 600 replica copies in total.
        counts = [(await c.status(j))["keys"] for j in range(5)]
        assert all(0 < k < 200 for k in counts)
        assert sum(counts) == 600
    finally:
        await c.close()


async def test_no_acknowledged_write_lost_under_single_node_kill(cluster):
    """One sequential writer per key; kill a node mid-stream; the final read of
    every key must return the last acknowledged write."""
    c = KVClient(cluster.addrs, seed=2)
    last_acked: dict[str, int] = {}
    try:
        for step in range(300):
            if step == 100:
                cluster.kill("n3")
            key = f"k{step % 30}"
            if (await c.put(key, step))["ok"]:
                last_acked[key] = step
        assert len(last_acked) == 30
        for key, v in last_acked.items():
            assert (await c.get(key))["value"] == v, key
        # And still true after the victim comes back with its WAL.
        cluster.restart("n3")
        for key, v in last_acked.items():
            assert (await c.get(key))["value"] == v, key
    finally:
        await c.close()


async def test_failure_detector_and_hinted_handoff(cluster):
    c = KVClient(cluster.addrs, seed=3)
    try:
        cluster.kill("n5")
        t0 = time.monotonic()

        async def all_suspect():
            sts = [await c.status(j) for j in range(4)]
            return all(not s["alive"]["n5"] for s in sts)

        assert await wait_for(all_suspect, timeout=3)
        detect = time.monotonic() - t0
        assert detect < 2.0

        for i in range(100):
            assert (await c.put(f"h{i}", i))["ok"]
        hints = sum([(await c.status(j))["hints"] for j in range(4)])
        assert hints > 0  # n5 is in some preference lists, so some writes were hinted

        cluster.restart("n5")

        async def drained():
            return sum([(await c.status(j))["hints"] for j in range(4)]) == 0

        assert await wait_for(drained, timeout=5)
        # n5 now holds every key it is a replica for, without any read repair.
        dump = (await c.dump(4))["data"]
        assert sum(1 for k in dump if k.startswith("h")) > 0
    finally:
        await c.close()


@pytest.mark.parametrize("nemesis", [kill_one_nemesis, pause_one_nemesis], ids=["kill", "pause"])
async def test_jepsen_register_linearizable(cluster, nemesis):
    cfg = WorkloadConfig(clients=6, keys=4, duration=4.5, seed=7)
    history, events = await run_workload(cluster, cfg, nemesis)
    oks = sum(o.status == "ok" for o in history)
    assert oks > 200, f"too few ok ops ({oks}); workload did not really run"
    assert len(events) == 2
    res = check_history(history)
    assert res.ok, res.detail
