import numpy as np
import pytest

from agentenv.metrics import bootstrap_ci, classify_failure, episodes_frame, pass_at_k, summarize


def test_pass_at_k_values():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    # 1 - C(7,2)/C(10,2) = 1 - 21/45
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - 21 / 45)
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)


def test_pass_at_k_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n, c, k = 8, 3, 4
    sims = [any(rng.permutation([1] * c + [0] * (n - c))[:k]) for _ in range(20000)]
    assert pass_at_k(n, c, k) == pytest.approx(np.mean(sims), abs=0.01)


def test_bootstrap_contains_mean():
    v = np.array([0, 1] * 50, dtype=float)
    lo, hi = bootstrap_ci(v)
    assert lo < 0.5 < hi and hi - lo < 0.25


def tr(**kw):
    base = {"task_id": "t", "family": "sql", "difficulty": "easy", "passed": False, "submitted": True,
            "stop_reason": "submitted", "steps": [], "verifier": {"meta": {}}}
    base.update(kw)
    return base


def test_taxonomy_rules():
    assert classify_failure(tr(passed=True)) is None
    assert classify_failure(tr(stop_reason="parse_errors", submitted=False)) == "format_errors"
    assert classify_failure(tr(stop_reason="token_budget", submitted=False)) == "token_budget"
    same = {"tool": "run_sql", "args": {"query": "x"}, "tool_ok": True}
    assert classify_failure(tr(submitted=False, stop_reason="step_limit", steps=[same] * 4)) == "looping_same_call"
    err = {"tool": "run_sql", "args": {"query": "y"}, "tool_ok": False}
    assert classify_failure(tr(submitted=False, stop_reason="step_limit", steps=[err, dict(err, args={"q": 1})])) \
        == "stuck_on_tool_errors"
    assert classify_failure(tr(submission="", verifier={"meta": {"reason": "unparseable"}})) \
        == "empty_or_unparseable_answer"
    assert classify_failure(tr(submission="3")) == "answered_without_query"
    ok = {"tool": "run_sql", "args": {}, "tool_ok": True}
    assert classify_failure(tr(submission="3", steps=[ok])) == "wrong_answer"
    assert classify_failure(tr(family="bugfix")) == "submitted_without_edit"
    ed = {"tool": "edit_file", "args": {}, "tool_ok": True}
    assert classify_failure(tr(family="bugfix", steps=[ed], verifier={"detail": "E   SyntaxError", "meta": {}})) \
        == "broke_import_or_syntax"
    assert classify_failure(tr(family="bugfix", steps=[ed], verifier={"detail": "1 failed", "meta": {}})) \
        == "tests_still_fail"


def test_summarize_groups():
    traces = []
    for t in range(4):
        for s in range(4):
            traces.append(tr(task_id=f"t{t}", sample=s, passed=(t < 2 and s < 2), difficulty="easy",
                             total_tokens=100, wall_s=1.0))
    df = episodes_frame(traces)
    summ = summarize(df, ks=(1, 2, 4), n_boot=200)
    row = summ[(summ.family == "sql") & (summ.difficulty == "all")].iloc[0]
    assert row["pass@1"] == pytest.approx(0.25)
    assert row["pass@4"] == pytest.approx(0.5)
    assert row["tokens_per_solved"] == pytest.approx(1600 / 4)
