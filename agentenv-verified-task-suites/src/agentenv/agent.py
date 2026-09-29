"""The agent loop: JSON tool calling with a step limit and a token budget.

Protocol. Each assistant turn must contain one JSON object
    {"tool": "<name>", "args": {...}}
(an optional "thought" key is allowed). The parser is tolerant of code fences
and prose around the object but takes only the first well formed call. The
tool result goes back as a user message that starts with "Tool result".

Budget. token_budget caps the total number of *generated* tokens in an
episode. The context the model reads is capped separately by
max_context_tokens. Both are recorded in the trace.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import Adapter
from .sandbox import Limits
from .task import Task, materialize
from .tools import Toolbox
from .verifiers import verify

SYSTEM_TEMPLATE = """You are an autonomous agent working in a sandboxed directory. You act only by calling tools.
Reply with exactly one JSON object per turn and nothing else, in this form
{{"tool": "<tool name>", "args": {{...}}}}

Available tools
{tools}

Rules
- One tool call per reply. Tool results come back in the next message.
- You have at most {steps} tool calls. Call submit when you are done.
- There is no network access."""


@dataclass
class EpisodeConfig:
    temperature: float = 0.0
    max_tokens_per_turn: int = 512
    max_context_tokens: int = 12000
    max_parse_errors: int = 3
    sandbox: Limits = Limits(cpu_s=20, wall_s=30, mem_mb=1024)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return (call, error). call is {"tool": str, "args": dict}."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for cand in candidates:
        i = cand.find("{")
        while i != -1:
            try:
                obj, _ = dec.raw_decode(cand, i)
            except json.JSONDecodeError:
                i = cand.find("{", i + 1)
                continue
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                args = obj.get("args", obj.get("arguments", {}))
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    return None, "args must be a JSON object"
                return {"tool": obj["tool"], "args": args}, None
            i = cand.find("{", i + 1)
    if "{" not in text:
        return None, "no JSON object found"
    return None, "no valid tool call JSON found"


def run_episode(task: Task, adapter: Adapter, seed: int, cfg: EpisodeConfig = EpisodeConfig(),
                sample: int = 0, keep_workspace: bool = False) -> dict[str, Any]:
    """Run one episode and return its full trace as a JSON-able dict."""
    t_start = time.monotonic()
    ws = Path(tempfile.mkdtemp(prefix=f"ep-{task.id}-"))
    trace: dict[str, Any] = {
        "task_id": task.id, "family": task.family, "difficulty": task.difficulty, "tags": task.tags,
        "adapter": adapter.name, "seed": seed, "sample": sample,
        "config": {"temperature": cfg.temperature, "max_tokens_per_turn": cfg.max_tokens_per_turn,
                   "step_limit": task.step_limit, "token_budget": task.token_budget,
                   "max_context_tokens": cfg.max_context_tokens},
        "steps": [],
    }
    prompt_tokens = completion_tokens = parse_errors = 0
    model_latency = 0.0
    submitted, submission, stop = False, None, "step_limit"
    try:
        materialize(task.fixtures, ws)
        tools = Toolbox(ws, task.allowed_tools, cfg.sandbox)
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(tools=tools.describe(), steps=task.step_limit)},
            {"role": "user", "content": task.prompt},
        ]
        adapter.start_episode(task, seed)
        for step in range(task.step_limit):
            if time.monotonic() - t_start > task.timeout_s:
                stop = "episode_timeout"
                break
            remaining = task.token_budget - completion_tokens
            if remaining <= 0:
                stop = "token_budget"
                break
            gen = adapter.generate(messages, max_tokens=min(cfg.max_tokens_per_turn, remaining),
                                   temperature=cfg.temperature, seed=seed * 1000 + step)
            prompt_tokens += gen.prompt_tokens
            completion_tokens += gen.completion_tokens
            model_latency += gen.latency_s
            rec: dict[str, Any] = {"step": step, "model_output": gen.text,
                                   "prompt_tokens": gen.prompt_tokens,
                                   "completion_tokens": gen.completion_tokens,
                                   "latency_s": round(gen.latency_s, 4),
                                   "finish_reason": gen.finish_reason}
            messages.append({"role": "assistant", "content": gen.text})
            call, err = parse_tool_call(gen.text)
            if call is None:
                parse_errors += 1
                obs = (f"Format error: {err}. Reply with one JSON object like "
                       '{"tool": "list_dir", "args": {"path": "."}}')
                rec.update(parse_error=err, observation=obs)
                trace["steps"].append(rec)
                messages.append({"role": "user", "content": obs})
                if parse_errors >= cfg.max_parse_errors:
                    stop = "parse_errors"
                    break
                continue
            t0 = time.monotonic()
            res = tools.call(call["tool"], call["args"])
            obs = f"Tool result ({call['tool']}):\n{res.output}"
            rec.update(tool=call["tool"], args=call["args"], observation=res.output, tool_ok=res.ok,
                       tool_meta=res.meta, tool_wall_s=round(time.monotonic() - t0, 4))
            trace["steps"].append(rec)
            if res.done:
                submitted, submission, stop = True, res.submission, "submitted"
                break
            messages.append({"role": "user", "content": obs})
            if gen.prompt_tokens > cfg.max_context_tokens:
                stop = "context_overflow"
                break
        vres = verify(task, ws, submission, submitted)
        trace["verifier"] = vres.to_dict()
        trace["passed"] = bool(vres.passed)
    except Exception as e:  # infrastructure failure, recorded not raised
        trace["verifier"] = {"passed": False, "detail": "harness error"}
        trace["passed"] = False
        trace["harness_error"] = "".join(traceback.format_exception(e))[-3000:]
        stop = "harness_error"
    finally:
        if not keep_workspace:
            shutil.rmtree(ws, ignore_errors=True)
    trace.update(
        stop_reason=stop, submitted=submitted, submission=submission,
        n_steps=len(trace["steps"]), parse_errors=parse_errors,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_latency_s=round(model_latency, 3),
        wall_s=round(time.monotonic() - t_start, 3),
    )
    return trace
