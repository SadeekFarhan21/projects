# Design

This document describes how the v0 harness is put together, the data structures it passes around, the invariants it relies on, and the trade-offs I chose and rejected.

## Data flow

```
                 scripts/download_data.sh
                 (Chinook, TPC-H via DuckDB dbgen, 3 packages at pinned tags)
                               |
                               v
  families/sql_family.py   families/bugfix_family.py + mutations.py
  (templates + ref SQL)    (AST mutants, keep only if hidden tests fail)
                \                 /
                 v               v
               tasks/{sql,bugfix}/*.yaml     <- Task (task.py), validated on load
                               |
                               v
  runner.py  run_eval(tasks, adapter_spec, n_samples, base_seed, workers)
      |  seed = sha256(base_seed:task_id:sample)  (independent of scheduling)
      |  ProcessPoolExecutor, one adapter per worker process
      v
  agent.py  run_episode(task, adapter, seed)
      |
      |  1. materialize fixtures into a fresh temp workspace
      |  2. loop until submit | step_limit | token_budget | parse_errors | timeout
      |        adapter.generate(messages) --> text
      |        parse_tool_call(text)      --> {"tool", "args"} or format error
      |        Toolbox.call(tool, args)   --> observation
      |              run_sql / run_python ----> sandbox.run_sandboxed
      |              read/write/edit/list  ----> path-confined file ops
      |  3. verify(task, workspace, submission)
      |        python verifier: numeric or text answer check
      |        command verifier: copy workspace + hidden files into a fresh
      |                          verify dir, run pytest in the sandbox
      v
  JSONL trace line (every model output, tool call, observation, token count,
  latency, stop reason, verifier result)
                               |
                               v
  metrics.py  pass@1, unbiased pass@k, bootstrap CI over tasks,
              tokens and wall seconds per solved episode, failure taxonomy
                               |
                               v
  scripts/analyze.py -> results/*.csv, *.json, *.png
```

Sandbox layering for every command an episode runs:

```
parent (runner worker)
  |- Popen(start_new_session=True)           process group, killpg on exit
  |    preexec: RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_NOFILE
  |    exec sandbox-exec -p <profile>        Seatbelt profile:
  |         (deny network*)
  |         (deny file-write*) except the workspace and /dev/null
  |         (deny file-read-data under /Users) except workspace and Python
  |    exec python ...                       env scrubbed, HOME and TMPDIR in workspace
  |- poll loop every 20 ms: wall clock, RSS of the whole process tree (psutil)
```

## Key data structures

### Task (task.py)

A dataclass that round-trips to YAML. Fields: `id`, `family`, `prompt`, `fixtures`, `allowed_tools`, `verifier`, `difficulty`, `tags`, `timeout_s`, `limits` (`step_limit`, `token_budget`), `reference`, `metadata`. `Task.from_dict` rejects unknown keys and validates tools, difficulty, verifier shape and fixture shape, so a malformed task fails at load time rather than in the middle of a run.

Fixtures are an ordered list of three kinds. `copy` copies a file or directory from the project (with exclude globs), `path` writes inline content, and `patch` replaces `old` with `new` at a character `offset` and fails if the text at that offset is not exactly `old`. The bug-fix tasks are "copy the pristine package, then patch one span", which keeps each YAML small (the whole 200 task set is under 1 MB) and makes the mutation explicit and checkable.

### Verifier spec

`{"type": "python", "function": "module:fn", "args": {...}}` calls `fn(VerifyContext, **args)`. `{"type": "command", "command": [...], "hidden": [fixtures], "timeout_s", "cpu_s", "mem_mb"}` runs the command in a fresh directory that holds a copy of the workspace plus the hidden fixtures.

### Trace (agent.py)

One JSON object per episode: task id, family, difficulty, adapter, seed, sample index, config, a `steps` list (model output, parsed tool call, observation, prompt and completion tokens, latency, tool wall time, tool errors), the verifier result, stop reason, submission and totals. Everything in `metrics.py` is computed from traces alone, so a run can be re-analyzed without re-running it.

### Adapter protocol (adapters.py)

`start_episode(task, seed)` and `generate(messages, max_tokens, temperature, seed) -> Generation(text, prompt_tokens, completion_tokens, latency_s, finish_reason)`. The null and reference baselines implement the same protocol and emit JSON tool calls, so they go through the same parser, tools, sandbox and verifier as a model.

## Invariants

1. Hidden tests never exist in an agent workspace. They are only materialized in the verify directory, and the Seatbelt profile denies reading anything under /Users outside the workspace and the Python install, so the agent cannot read `data/` or `tasks/` from inside `run_python` either.
2. Every task is known solvable. The reference agent passes every task on every repeat, through the normal tool path (validated by `scripts/validate_verifiers.py`).
3. Every task is non-trivial. The null agent (submit immediately, empty answer) fails every task on every repeat. For SQL tasks the generator also rejects answers equal to zero, so submitting "0" is not a free pass.
4. Each bug-fix task differs from the pristine package by exactly one span, and restoring that span makes the hidden tests pass.
5. Episode seeds depend only on (base_seed, task_id, sample). Changing the worker count or the task subset does not change the seed of any job.
6. Harness failures are recorded, not raised. A crash inside an episode becomes `stop_reason = harness_error` and `passed = false` in the trace, so one bad task cannot kill a run and cannot be silently dropped.
7. Text answers have a unique ground truth. Argmax templates are rejected when the top two rows tie.

## Trade-offs

### Chosen

- Mutation testing over hand-written bugs. It gives 100 bugs in about 20 minutes of compute with a known fix (the inverse edit), and the "keep only if a hidden test fails" filter removes equivalent mutants automatically. The cost is that bugs are syntactically small and not always realistic.
- Hidden tests scoped to the mutated module for toolz (its test file for that module), the whole suite for inflection and semver. Running only the relevant files keeps a verification near 3 to 5 seconds on the loaded machine. The risk is that a fix breaks a different module and still passes. I accepted that because a single file edit in toolz rarely affects other modules' tests, and I list a full-suite check as a follow up.
- Seatbelt plus a polling memory watchdog. macOS rejects RLIMIT_AS and RLIMIT_DATA, so a pure rlimit sandbox cannot cap memory there. Polling RSS every 50 ms can overshoot briefly, which is fine for accidental blowups and not a defense against an adversary.
- A tolerant JSON tool-call parser (code fences, prose around the object, `arguments` alias) instead of a model-specific tool format. It works with any chat model, including the API adapter, and parse failures are counted as their own failure category instead of being hidden.
- An extra `edit_file` tool beyond the spec list. A 1.5B model cannot rewrite a 1000 line file inside a 4000 token budget, so without a search-and-replace edit the bug-fix family would be unsolvable by construction for small models.
- The compact schema in SQL prompts. In the first smoke test the model guessed table names (`tracks`, `genre_id`) and looped on the same failing query. Giving `Table(col, ...)` lines is standard in text-to-SQL benchmarks and moves the task from "guess the schema" to "write the query".
- Bootstrap over tasks, not over episodes. Samples of one task are correlated, so resampling episodes would give intervals that are too narrow.

### Rejected

- Docker or a VM per episode. Heavier, slower to start, and not needed to demonstrate the harness on one Mac. Seatbelt gives network and filesystem isolation at process start cost.
- Running tools in threads inside the worker. A runaway `run_python` could hang or exhaust the worker; a subprocess with a process group can always be killed.
- Letting the verifier read the agent's workspace in place. The verify directory is a copy, so an agent cannot leave a background process that edits files during verification, and symlinks are copied as links (not followed) so they cannot pull in files from outside.
- Batching generation across episodes with one model. It would be faster, but episodes are at different turns with different prompt lengths. One model per worker process is simpler and keeps episodes independent; batching is a later optimization.
- Exact string match for numeric answers. Floating point sums and "give 2 decimals" answers need tolerance; I use rel 1e-4 or abs 0.01.
