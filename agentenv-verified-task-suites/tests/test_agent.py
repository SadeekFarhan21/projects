import json

from agentenv.adapters import Generation
from agentenv.agent import EpisodeConfig, parse_tool_call, run_episode
from agentenv.task import Task


def test_parse_variants():
    c, e = parse_tool_call('{"tool": "submit", "args": {"answer": 3}}')
    assert c == {"tool": "submit", "args": {"answer": 3}} and e is None
    c, _ = parse_tool_call('Sure!\n```json\n{"tool": "list_dir", "args": {"path": "."}}\n```')
    assert c["tool"] == "list_dir"
    c, _ = parse_tool_call('I think {not json} then {"thought": "x", "tool": "submit", "arguments": {}}')
    assert c == {"tool": "submit", "args": {}}
    c, _ = parse_tool_call('{"tool": "run_sql", "args": {"query": "SELECT \'{a}\'"}}')
    assert c["args"]["query"] == "SELECT '{a}'"
    assert parse_tool_call("no json here")[0] is None
    assert parse_tool_call('{"tool": "x", "args": [1]}')[1] == "args must be a JSON object"


class Scripted:
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def start_episode(self, task, seed):
        pass

    def generate(self, messages, *, max_tokens, temperature, seed):
        out = self.outputs.pop(0) if self.outputs else "..."
        return Generation(out, 10, 50, 0.0)


def task(**kw):
    d = dict(id="t", family="sql", prompt="what is 2+2", fixtures=[{"path": "a.txt", "content": "hi"}],
             allowed_tools=["read_file", "submit"],
             verifier={"type": "python", "function": "agentenv.verifiers:numeric_or_text_answer",
                       "args": {"expected": 4}}, limits={"step_limit": 4, "token_budget": 1000})
    d.update(kw)
    return Task.from_dict(d)


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def test_episode_pass_and_trace():
    tr = run_episode(task(), Scripted([call("read_file", path="a.txt"), call("submit", answer="4")]), seed=1)
    assert tr["passed"] and tr["stop_reason"] == "submitted" and tr["n_steps"] == 2
    assert "hi" in tr["steps"][0]["observation"]
    assert tr["completion_tokens"] == 100
    json.dumps(tr)  # trace is JSON serializable


def test_step_limit_and_parse_errors_and_budget():
    tr = run_episode(task(), Scripted([call("read_file", path="a.txt")] * 10), seed=1)
    assert tr["stop_reason"] == "step_limit" and not tr["passed"] and tr["n_steps"] == 4
    tr = run_episode(task(), Scripted(["garbage"] * 10), seed=1, cfg=EpisodeConfig(max_parse_errors=2))
    assert tr["stop_reason"] == "parse_errors" and tr["parse_errors"] == 2
    tr = run_episode(task(limits={"step_limit": 10, "token_budget": 120}),
                     Scripted([call("read_file", path="a.txt")] * 10), seed=1)
    assert tr["stop_reason"] == "token_budget" and tr["completion_tokens"] == 150


def test_harness_error_is_recorded():
    t = task(fixtures=[{"copy": "/nonexistent/file"}])
    tr = run_episode(t, Scripted([]), seed=1)
    assert tr["stop_reason"] == "harness_error" and not tr["passed"] and "harness_error" in tr
