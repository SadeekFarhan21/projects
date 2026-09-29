import sqlite3

from agentenv.tools import Toolbox

ALL = ["run_sql", "run_python", "read_file", "write_file", "edit_file", "list_dir", "submit"]


def box(ws, allowed=ALL):
    return Toolbox(ws, allowed)


def test_file_tools_and_escape(ws):
    b = box(ws)
    assert b.call("write_file", {"path": "a/b.py", "content": "x = 1\ny = 2\n"}).ok
    r = b.call("read_file", {"path": "a/b.py", "start_line": 2})
    assert "y = 2" in r.output and "x = 1" not in r.output
    assert "b.py" in b.call("list_dir", {"path": "a"}).output
    for bad in ("../x", "/etc/passwd", "a/../../x"):
        r = b.call("read_file", {"path": bad})
        assert not r.ok and "outside" in r.output


def test_symlink_escape_blocked(ws, tmp_path):
    (tmp_path / "secret.txt").write_text("s")
    (ws / "link").symlink_to(tmp_path / "secret.txt")
    r = box(ws).call("read_file", {"path": "link"})
    assert not r.ok


def test_edit_file_requires_unique_match(ws):
    b = box(ws)
    b.call("write_file", {"path": "f.py", "content": "a = 1\na = 1\nb = 2\n"})
    assert not b.call("edit_file", {"path": "f.py", "old": "a = 1", "new": "a = 3"}).ok
    assert b.call("edit_file", {"path": "f.py", "old": "b = 2", "new": "b = 3"}).ok
    assert (ws / "f.py").read_text().endswith("b = 3\n")


def test_run_sql_is_read_only(ws):
    con = sqlite3.connect(ws / "db.sqlite")
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(40)])
    con.commit()
    con.close()
    b = box(ws)
    r = b.call("run_sql", {"query": "SELECT SUM(x) FROM t"})
    assert r.ok and "780" in r.output
    r = b.call("run_sql", {"query": "SELECT x FROM t"})
    assert "(more rows)" in r.output
    r = b.call("run_sql", {"query": "DELETE FROM t"})
    assert not r.ok and "readonly" in r.output.lower()


def test_run_python_and_dispatch_errors(ws):
    b = box(ws, ["run_python", "submit"])
    r = b.call("run_python", {"code": "print(6 * 7)"})
    assert r.ok and "42" in r.output
    assert b.call("read_file", {"path": "x"}).meta["error"] == "disallowed_tool"
    assert b.call("nope", {}).meta["error"] == "unknown_tool"
    assert b.call("run_python", {"cod": "1"}).meta["error"] == "bad_args"
    s = b.call("submit", {"answer": 5})
    assert s.done and s.submission == 5
