"""In-process tests: several Node objects on one event loop, real TCP between them."""

import asyncio

import pytest

from kvstore.client import KVClient
from kvstore.cluster import free_ports
from kvstore.node import Node, NodeConfig
from kvstore.scenarios import run_inversion_scenario


def make_cfgs(tmp_path, size=3, n=3, r=2, w=2):
    ids = [f"n{i + 1}" for i in range(size)]
    peers = {i: ("127.0.0.1", p) for i, p in zip(ids, free_ports(size))}
    return [
        NodeConfig(node_id=i, host="127.0.0.1", port=peers[i][1], peers=peers, n=n, r=r, w=w,
                   data_dir=str(tmp_path / i), heartbeat_interval=0.05, suspect_after=0.25,
                   rpc_timeout=0.5)
        for i in ids
    ]


async def start_all(cfgs):
    nodes = [Node(c) for c in cfgs]
    for nd in nodes:
        await nd.start()
    return nodes


async def stop_all(nodes):
    for nd in nodes:
        await nd.stop()


def test_quorum_config_validation(tmp_path):
    cfg = make_cfgs(tmp_path)[0]
    cfg.r = 4
    with pytest.raises(ValueError):
        cfg.validate()


async def test_put_get_roundtrip(tmp_path):
    nodes = await start_all(make_cfgs(tmp_path))
    c = KVClient([("127.0.0.1", nd.cfg.port) for nd in nodes])
    try:
        assert (await c.get("missing"))["value"] is None
        assert (await c.put("a", 1))["ok"]
        assert (await c.get("a"))["value"] == 1
        assert (await c.put("a", {"nested": [1, 2]}))["ok"]
        assert (await c.get("a"))["value"] == {"nested": [1, 2]}
        # Every replica got the write (N=3 on a 3-node cluster) eventually.
        await asyncio.sleep(0.05)
        assert all(nd.store.get("a")[1] == {"nested": [1, 2]} for nd in nodes)
    finally:
        await c.close()
        await stop_all(nodes)


async def test_versions_increase_across_coordinators(tmp_path):
    nodes = await start_all(make_cfgs(tmp_path))
    try:
        v1 = (await nodes[0].coord_put("k", "x"))["ver"]
        v2 = (await nodes[1].coord_put("k", "y"))["ver"]
        v3 = (await nodes[2].coord_put("k", "z"))["ver"]
        assert tuple(v1) < tuple(v2) < tuple(v3)
        assert (await nodes[0].coord_get("k"))["value"] == "z"
    finally:
        await stop_all(nodes)


async def test_read_writeback_repairs_to_write_quorum(tmp_path):
    cfgs = make_cfgs(tmp_path, r=1, w=3)
    nodes = await start_all(cfgs)
    try:
        await nodes[0].coord_put("k", "old")
        # Simulate a write that reached only one replica before its coordinator died.
        nodes[2].store.apply("k", (99, "zz", 1), "new")
        # With R=1 the coordinator's own replica answers first (local short-circuit),
        # so a read coordinated by n3 sees "new" and must push it to W=3 replicas
        # before returning it.
        res = await nodes[2].coord_get("k")
        assert res["value"] == "new"
        assert nodes[2].stats.read_writebacks == 1
        assert all(nd.store.get("k")[1] == "new" for nd in nodes)
        # Any later read, from any coordinator, now sees it too.
        for nd in nodes:
            assert (await nd.coord_get("k"))["value"] == "new"
    finally:
        await stop_all(nodes)


async def test_write_survives_one_down_replica_and_hint_is_delivered(tmp_path):
    cfgs = make_cfgs(tmp_path)
    nodes = await start_all(cfgs)
    try:
        await nodes[2].stop()  # crash n3
        await asyncio.sleep(0.4)  # let the detector notice
        assert not nodes[0].alive("n3")
        res = await nodes[0].coord_put("k", "v")
        assert res["ok"]
        assert nodes[0].hints["n3"]["k"][1] == "v"
        # Bring n3 back with the same data dir (WAL replay), then wait for handoff.
        nodes[2] = Node(cfgs[2])
        await nodes[2].start()
        for _ in range(50):
            await asyncio.sleep(0.1)
            if nodes[2].store.get("k")[1] == "v" and not nodes[0].hints["n3"]:
                break
        assert nodes[2].store.get("k")[1] == "v"
        assert nodes[0].hints["n3"] == {}
        assert nodes[0].stats.hints_delivered >= 1
    finally:
        await stop_all(nodes)


async def test_unavailable_when_quorum_impossible(tmp_path):
    nodes = await start_all(make_cfgs(tmp_path))
    try:
        await nodes[1].stop()
        await nodes[2].stop()
        res = await nodes[0].coord_put("k", 1)
        assert not res["ok"] and res["definite"]  # phase 1 failed, nothing written
        res = await nodes[0].coord_get("k")
        assert not res["ok"]
    finally:
        await nodes[0].stop()


async def test_wal_replay_after_restart(tmp_path):
    cfgs = make_cfgs(tmp_path, size=1, n=1, r=1, w=1)
    nodes = await start_all(cfgs)
    for i in range(20):
        await nodes[0].coord_put(f"k{i}", i)
    await nodes[0].stop()
    again = Node(cfgs[0])
    assert all(again.store.get(f"k{i}")[1] == i for i in range(20))
    again.store.close()


async def test_without_writeback_new_old_inversion_is_caught(tmp_path):
    r1, r2, linearizable = await run_inversion_scenario(str(tmp_path), writeback=False)
    assert (r1, r2) == ("new", "old")
    assert not linearizable


async def test_with_writeback_same_scenario_is_linearizable(tmp_path):
    r1, r2, linearizable = await run_inversion_scenario(str(tmp_path), writeback=True)
    assert r1 == "new" and r2 == "new"
    assert linearizable
