"""Integration tests on the generated task files (need data/ and tasks/)."""
import json

from agentenv import PROJECT_ROOT
from agentenv.adapters import NullAdapter, ReferenceAdapter
from agentenv.agent import run_episode
from agentenv.runner import episode_seed, run_eval
from agentenv.task import load_tasks

from conftest import needs_data


@needs_data
def test_family_sizes_and_balance():
    tasks = load_tasks(PROJECT_ROOT / "tasks")
    fams = {f: [t for t in tasks if t.family == f] for f in ("sql", "bugfix")}
    assert len(fams["sql"]) == 100
    assert len(fams["bugfix"]) == 100
    for f, ts in fams.items():
        assert {t.difficulty for t in ts} == {"easy", "medium", "hard"}
    assert {t.metadata["operator"] for t in fams["bugfix"]} == {
        "flip_comparison", "off_by_one", "wrong_variable", "dropped_branch"}


@needs_data
def test_sql_reference_passes_and_null_fails():
    tasks = load_tasks(PROJECT_ROOT / "tasks", family="sql")[:3]
    for t in tasks:
        assert run_episode(t, ReferenceAdapter(), seed=0)["passed"]
        assert not run_episode(t, NullAdapter(), seed=0)["passed"]


@needs_data
def test_bugfix_reference_passes_and_null_fails():
    t = load_tasks(PROJECT_ROOT / "tasks", family="bugfix")[0]
    ref = run_episode(t, ReferenceAdapter(), seed=0)
    assert ref["passed"], ref["verifier"]["detail"]
    null = run_episode(t, NullAdapter(), seed=0)
    assert not null["passed"] and null["verifier"]["meta"]["failing"]


def test_episode_seed_is_stable():
    assert episode_seed(0, "a", 0) == episode_seed(0, "a", 0)
    assert len({episode_seed(0, "a", s) for s in range(50)}) == 50
    assert episode_seed(0, "a", 0) != episode_seed(1, "a", 0)


@needs_data
def test_parallel_runner_writes_jsonl(tmp_path):
    tasks = load_tasks(PROJECT_ROOT / "tasks", family="sql")[:4]
    out = tmp_path / "run.jsonl"
    meta = run_eval(tasks, "reference", out, n_samples=2, workers=2, progress=False)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 8 and meta["passed"] == 8
    assert {(x["task_id"], x["sample"]) for x in lines} == {(t.id, s) for t in tasks for s in range(2)}
    assert all(x["seed"] == episode_seed(0, x["task_id"], x["sample"]) for x in lines)


@needs_data
def test_reference_revert_edit_is_exact_and_small():
    from agentenv.adapters import revert_edit
    for t in load_tasks(PROJECT_ROOT / "tasks", family="bugfix"):
        r = t.reference["restore"]
        pristine = (PROJECT_ROOT / r["from"]).read_text()
        (p,) = [f for f in t.fixtures if f.get("patch") == r["path"]]
        mutated = pristine[:p["offset"]] + p["new"] + pristine[p["offset"] + len(p["old"]):]
        old, new = revert_edit(pristine, p)
        assert mutated.count(old) == 1
        assert mutated.replace(old, new, 1) == pristine
        assert len(old) < 2000
