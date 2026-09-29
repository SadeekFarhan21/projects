"""Model adapters. Every policy, including the scripted baselines, speaks the
same interface: chat messages in, text out, plus token counts. Scripted
policies emit the same JSON tool calls a model would, so they exercise the
exact parser, tool and verifier path that model episodes use.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import PROJECT_ROOT
from .task import Task


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str = "stop"


class Adapter(Protocol):
    name: str

    def start_episode(self, task: Task, seed: int) -> None: ...

    def generate(self, messages: list[dict[str, str]], *, max_tokens: int,
                 temperature: float, seed: int) -> Generation: ...


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _call(tool: str, **args: Any) -> str:
    return json.dumps({"tool": tool, "args": args})


def revert_edit(pristine: str, patch: dict) -> tuple[str, str]:
    """Smallest (mutated_snippet, pristine_snippet) pair, grown line by line
    around the patch, whose mutated snippet occurs exactly once in the mutated
    file. Applying it with edit_file restores the pristine file exactly."""
    off, old, new = int(patch["offset"]), patch["old"], patch["new"]
    mutated = pristine[:off] + new + pristine[off + len(old):]
    lo, hi = off, off + len(new)
    while True:
        a = mutated.rfind("\n", 0, lo) + 1 if lo > 0 else 0
        b = mutated.find("\n", hi)
        b = len(mutated) if b == -1 else b + 1
        snippet = mutated[a:b]
        if mutated.count(snippet) == 1 or (a == 0 and b == len(mutated)):
            restored = snippet[: off - a] + old + snippet[off - a + len(new):]
            return snippet, restored
        lo, hi = max(0, a - 1), min(len(mutated), b + 1)


# ---- scripted baselines ------------------------------------------------------

class NullAdapter:
    """Submits immediately with an empty answer. Every verifier must fail it."""

    name = "null"

    def start_episode(self, task: Task, seed: int) -> None:
        pass

    def generate(self, messages, *, max_tokens, temperature, seed) -> Generation:
        return Generation(_call("submit", answer=""), sum(_approx_tokens(m["content"]) for m in messages), 8, 0.0)


class ReferenceAdapter:
    """Replays the task's reference solution through the normal tool interface.

    sql tasks: run the reference SQL with run_sql, then submit the first cell of
    the first row as it came back from the tool (so the tool path is tested).
    bugfix tasks: write the pristine file back (reverting the mutation), submit.
    """

    name = "reference"

    def __init__(self) -> None:
        self.plan: list[str] = []
        self.task: Task | None = None

    def start_episode(self, task: Task, seed: int) -> None:
        self.task = task
        ref = task.reference
        if "sql" in ref:
            self.plan = [_call("run_sql", query=ref["sql"]), "__SUBMIT_FIRST_CELL__"]
        elif "restore" in ref:
            r = ref["restore"]
            pristine = (PROJECT_ROOT / r["from"]).read_text()
            patches = [f for f in task.fixtures if f.get("patch") == r["path"]]
            if len(patches) == 1:
                old, new = revert_edit(pristine, patches[0])
                first = _call("edit_file", path=r["path"], old=old, new=new)
            else:  # no single patch to invert, fall back to rewriting the file
                first = _call("write_file", path=r["path"], content=pristine)
            self.plan = [first, _call("submit", answer="")]
        else:
            raise ValueError(f"{task.id}: no reference solution")

    def generate(self, messages, *, max_tokens, temperature, seed) -> Generation:
        step = self.plan.pop(0)
        if step == "__SUBMIT_FIRST_CELL__":
            obs = messages[-1]["content"]
            # observation format: "Tool result (run_sql):\n<header>\n<row1>..." (see agent.py)
            lines = obs.split("\n")
            row = lines[2] if len(lines) > 2 else ""
            step = _call("submit", answer=row.split(" | ")[0])
        return Generation(step, sum(_approx_tokens(m["content"]) for m in messages),
                          _approx_tokens(step), 0.0)


# ---- MLX local model -----------------------------------------------------------

class MLXAdapter:
    """Local model through mlx-lm. One instance per worker process."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                 top_p: float = 0.95):
        from mlx_lm import load

        self.model_id = model_id
        self.name = f"mlx:{model_id}"
        self.model, self.tokenizer = load(model_id)
        self.top_p = top_p

    def start_episode(self, task: Task, seed: int) -> None:
        pass

    def generate(self, messages, *, max_tokens, temperature, seed) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        sampler = make_sampler(temp=temperature, top_p=self.top_p if temperature > 0 else 0.0)
        t0 = time.monotonic()
        text, last = "", None
        for resp in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens,
                                    sampler=sampler):
            text += resp.text
            last = resp
        latency = time.monotonic() - t0
        return Generation(
            text=text,
            prompt_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(last, "generation_tokens", 0) or 0),
            latency_s=latency,
            finish_reason=str(getattr(last, "finish_reason", "stop") or "stop"),
        )


# ---- API adapter (OpenAI compatible chat completions) -----------------------------

class APIAdapter:
    """Adapter for any OpenAI compatible /v1/chat/completions endpoint.

    It never talks to a paid API unless the caller passes a base_url and sets
    AGENTENV_ALLOW_API=1. The tests point it at a local stub server. Nothing
    in this repository sets that variable.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 timeout_s: float = 60.0):
        self.model = model
        self.name = f"api:{model}"
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    def start_episode(self, task: Task, seed: int) -> None:
        pass

    def build_request(self, messages, *, max_tokens, temperature, seed) -> dict[str, Any]:
        return {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed}

    def generate(self, messages, *, max_tokens, temperature, seed) -> Generation:
        if not self.base_url:
            raise RuntimeError("APIAdapter has no base_url; refusing to guess an endpoint")
        is_local = self.base_url.startswith(("http://127.0.0.1", "http://localhost"))
        if not is_local and os.environ.get("AGENTENV_ALLOW_API") != "1":
            raise RuntimeError("remote API calls are disabled (set AGENTENV_ALLOW_API=1 to enable)")
        body = json.dumps(self.build_request(messages, max_tokens=max_tokens,
                                             temperature=temperature, seed=seed)).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", body, headers)
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read())
        usage = data.get("usage", {})
        choice = data["choices"][0]
        return Generation(choice["message"]["content"], int(usage.get("prompt_tokens", 0)),
                          int(usage.get("completion_tokens", 0)), time.monotonic() - t0,
                          choice.get("finish_reason", "stop"))


def make_adapter(spec: str) -> Adapter:
    """spec: 'null' | 'reference' | 'mlx:<hf id>' | 'api:<model>@<base_url>'."""
    if spec == "null":
        return NullAdapter()
    if spec == "reference":
        return ReferenceAdapter()
    if spec.startswith("mlx:"):
        return MLXAdapter(spec[4:])
    if spec.startswith("api:"):
        model, _, url = spec[4:].partition("@")
        return APIAdapter(model, base_url=url or None)
    raise ValueError(f"unknown adapter spec {spec!r}")


__all__ = ["Adapter", "Generation", "NullAdapter", "ReferenceAdapter", "MLXAdapter", "APIAdapter",
           "make_adapter", "Path"]
