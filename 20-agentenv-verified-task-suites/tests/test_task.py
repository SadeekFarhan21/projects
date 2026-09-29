import pytest

from agentenv.task import Task, TaskError, materialize


def make(**kw):
    d = dict(id="t1", family="sql", prompt="p", fixtures=[], allowed_tools=["submit"],
             verifier={"type": "python", "function": "agentenv.verifiers:numeric_or_text_answer",
                       "args": {"expected": 1}})
    d.update(kw)
    return Task.from_dict(d)


def test_yaml_round_trip(tmp_path):
    t = make(prompt="line one\nline two", tags=["a"], difficulty="hard", limits={"step_limit": 3})
    t.save(tmp_path / "t.yaml")
    t2 = Task.load(tmp_path / "t.yaml")
    assert t2 == t
    assert t2.step_limit == 3 and t2.token_budget == 4000
    assert "|" in (tmp_path / "t.yaml").read_text()  # multi-line prompt as a block scalar


@pytest.mark.parametrize("kw", [
    {"difficulty": "extreme"},
    {"allowed_tools": ["submit", "rm_rf"]},
    {"allowed_tools": ["run_sql"]},
    {"verifier": {"type": "magic"}},
    {"verifier": {"type": "command"}},
    {"fixtures": [{"copy": "a", "path": "b"}]},
    {"timeout_s": 0},
])
def test_validation_rejects(kw):
    with pytest.raises(TaskError):
        make(**kw)


def test_unknown_keys_rejected():
    with pytest.raises(TaskError):
        Task.from_dict({"id": "x", "bogus": 1})


def test_materialize_copy_inline_patch(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "m.py").write_text("x = 1 < 2\n")
    (src / "pkg" / "__pycache__").mkdir()
    ws = tmp_path / "ws"
    materialize([
        {"copy": str(src / "pkg"), "to": "pkg"},
        {"path": "notes/a.txt", "content": "hi"},
        {"patch": "pkg/m.py", "offset": 6, "old": "<", "new": ">="},
    ], ws)
    assert (ws / "pkg" / "m.py").read_text() == "x = 1 >= 2\n"
    assert (ws / "notes" / "a.txt").read_text() == "hi"
    assert not (ws / "pkg" / "__pycache__").exists()


def test_patch_mismatch_and_escape(tmp_path):
    ws = tmp_path / "ws"
    materialize([{"path": "a.py", "content": "abc"}], ws)
    with pytest.raises(TaskError):
        materialize([{"patch": "a.py", "offset": 0, "old": "zz", "new": "y"}], ws)
    with pytest.raises(TaskError):
        materialize([{"path": "../evil.txt", "content": "x"}], ws)
    with pytest.raises(TaskError):
        materialize([{"copy": str(tmp_path / "missing")}], ws)
