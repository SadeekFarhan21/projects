import numpy as np
import pytest

from marketsim.engine import Fundamental, Scheduler


def test_events_run_in_time_then_insertion_order():
    s, log = Scheduler(), []
    s.at(2.0, log.append, "c")
    s.at(1.0, log.append, "a")
    s.at(1.0, log.append, "b")  # same time: scheduled later, runs later
    s.run(10.0)
    assert log == ["a", "b", "c"] and s.t == 10.0 and s.n_events == 3


def test_run_stops_at_t_end_and_can_resume():
    s, log = Scheduler(), []
    s.at(1.0, log.append, 1)
    s.at(5.0, log.append, 5)
    s.run(3.0)
    assert log == [1]
    s.run(6.0)
    assert log == [1, 5]


def test_zero_delay_events_run_after_current_event():
    s, log = Scheduler(), []

    def first():
        s.after(0.0, log.append, "reaction")
        log.append("first")

    s.at(1.0, first)
    s.at(1.0, log.append, "second")
    s.run(2.0)
    # "second" was queued before the reaction, so it runs first.
    assert log == ["first", "second", "reaction"]


def test_cannot_schedule_in_past():
    s = Scheduler()
    s.run(5.0)
    with pytest.raises(ValueError):
        s.at(4.0, print)


def test_fundamental_increment_variance():
    # Var of one-step increments = sigma^2 dt + jump_rate dt jump_sd^2.
    s = Scheduler()
    f = Fundamental(s, np.random.default_rng(0), 0.0, sigma=1.0, jump_rate=0.05, jump_sd=4.0, dt=1.0)
    vals = []
    for t in range(1, 40_001):
        s.run(t + 0.5)
        vals.append(f.value)
    inc = np.diff(vals)
    expected = 1.0 + 0.05 * 16.0
    assert abs(inc.var() / expected - 1) < 0.05
    assert abs(f.n_jumps / 40_000 - 0.05) < 0.005
