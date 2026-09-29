import math

from kvstore.checker import Op, check_history, check_register


def w(p, v, a, b, key="x", status="ok"):
    return Op(p, "write", key, v, a, b if status != "info" else math.inf, status)


def r(p, v, a, b, key="x", status="ok"):
    return Op(p, "read", key, v, a, b, status)


def test_empty_and_sequential():
    assert check_register([]).ok
    assert check_register([w(0, 1, 0, 1), r(1, 1, 2, 3), w(0, 2, 4, 5), r(1, 2, 6, 7)]).ok


def test_read_initial_value_none():
    assert check_register([r(0, None, 0, 1), w(0, 1, 2, 3)]).ok
    assert not check_register([w(0, 1, 0, 1), r(1, None, 2, 3)]).ok


def test_stale_read_after_completed_write_is_rejected():
    h = [w(0, 1, 0, 1), w(0, 2, 2, 3), r(1, 1, 4, 5)]
    assert not check_register(h).ok


def test_concurrent_write_either_order_ok():
    # Write 2 overlaps both reads, so reads may see 1 then 2.
    h = [w(0, 1, 0, 1), w(0, 2, 2, 10), r(1, 1, 3, 4), r(2, 2, 5, 6)]
    assert check_register(h).ok


def test_new_old_inversion_rejected():
    # The classic non-atomic register anomaly: while write 2 is in flight,
    # one read sees 2 and a strictly later read sees 1.
    h = [w(0, 1, 0, 1), w(0, 2, 2, 10), r(1, 2, 3, 4), r(2, 1, 5, 6)]
    assert not check_register(h).ok


def test_failed_write_is_ignored_info_write_is_optional():
    assert check_register([w(0, 1, 0, 1, status="fail"), r(1, None, 2, 3)]).ok
    # Info write may never happen...
    assert check_register([w(0, 1, 0, 1, status="info"), r(1, None, 2, 3)]).ok
    # ...or happen long after it "returned".
    assert check_register([w(0, 1, 0, 1, status="info"), r(1, None, 2, 3), r(1, 1, 50, 51)]).ok
    # But it cannot un-happen once observed.
    assert not check_register(
        [w(0, 1, 0, 1, status="info"), r(1, 1, 2, 3), r(1, None, 4, 5)]
    ).ok


def test_read_of_never_written_value_rejected():
    assert not check_register([w(0, 1, 0, 1), r(1, 99, 2, 3)]).ok


def test_multikey_is_checked_per_key():
    h = [w(0, 1, 0, 1, key="a"), w(0, 5, 0, 1, key="b"), r(1, 1, 2, 3, key="a"), r(1, 5, 2, 3, key="b")]
    assert check_history(h).ok
    h.append(r(2, 1, 4, 5, key="b"))
    res = check_history(h)
    assert not res.ok and res.key == "b"


def test_many_concurrent_ops_stays_fast():
    # 5 processes, 40 rounds, heavy overlap: the memo keeps this tractable.
    ops = []
    t = 0.0
    last = None
    for rnd in range(40):
        v = rnd + 1
        ops.append(w(0, v, t, t + 3))
        for p in range(1, 5):
            ops.append(r(p, v if p % 2 else last, t + 0.5, t + 2.5))
        last = v
        t += 3.5
    res = check_register(ops)
    assert res.ok
    assert res.states_explored < 100_000
