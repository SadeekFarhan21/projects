"""Reproducible measurements of DiffSense's non-Claude analysis paths.

Three measurements, no network APIs, no Claude:
  1. drift   : hybrid CodeBERT+MiniLM drift scores over the commits of a git repo
  2. planted : breaking-change detector on a synthetic repo with known labels
  3. history : breaking-change detector on every commit of a real repo

Run from the repo root with backend/ importable:
  python scripts/measure.py drift   <repo> <out.json> [--model microsoft/codebert-base|all-MiniLM-L6-v2]
  python scripts/measure.py planted <out.json>
  python scripts/measure.py history <repo> <out.json>
"""
import json, os, sys, subprocess, tempfile, logging, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
logging.disable(logging.CRITICAL)
import numpy as np
import git


def commits_oldest_first(repo):
    return [c for c in reversed(list(repo.iter_commits("HEAD"))) if len(c.parents) <= 1]


def drift(repo_path, out, model):
    from src.embedding_engine import SemanticAnalyzer
    from src.git_analyzer import GitAnalyzer
    os.environ["CODE_MODEL"] = model
    ga = GitAnalyzer(repo_path)
    sa = SemanticAnalyzer()
    rows, embs = [], []
    for c in commits_oldest_first(ga.repo):
        # NOTE: GitAnalyzer.extract_diff_info() calls commit.diff(parent) without create_patch=True and
        # returns empty line lists, so we build the patch the way main.py / the detector do.
        parent = c.parents[0] if c.parents else None
        diffs = parent.diff(c, create_patch=True) if parent else c.diff(git.NULL_TREE, create_patch=True)
        infos, added, removed = [], [], []
        for df in diffs:
            path = df.b_path or df.a_path
            if not path or not path.endswith((".py", ".js", ".jsx", ".ts", ".tsx")) or not df.diff:
                continue
            a, rm = ga._parse_diff_lines(df.diff.decode("utf-8", errors="ignore"))
            if parent is None:
                a, rm = rm, a  # NULL_TREE diff is reversed
            infos.append(path); added += a; removed += rm
        if not infos:
            continue
        r = sa.generate_hybrid_embedding((added, removed), c.message.strip())
        embs.append(r.embedding)
        rows.append(dict(commit=c.hexsha[:7], author=c.author.name, msg=c.message.strip().split("\n")[0][:70],
                         files=len(infos), added=len(added), removed=len(removed)))
    for i, r in enumerate(rows):
        r["drift_vs_prev"] = None if i == 0 else round(1 - float(sa.calculate_similarity(embs[i-1], embs[i])), 4)
        r["drift_vs_first"] = None if i == 0 else round(1 - float(sa.calculate_similarity(embs[0], embs[i])), 4)
    vp = [r["drift_vs_prev"] for r in rows if r["drift_vs_prev"] is not None]
    summary = dict(model=model, n_commits=len(rows), threshold=0.3,
                   n_over_threshold_vs_prev=sum(v > 0.3 for v in vp),
                   drift_vs_prev_min=min(vp), drift_vs_prev_median=float(np.median(vp)), drift_vs_prev_max=max(vp),
                   embedding_dim=int(len(embs[0])))
    json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


def run(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "GIT_AUTHOR_DATE": "2025-01-01T00:00:00", "GIT_COMMITTER_DATE": "2025-01-01T00:00:00"})


BASE_PY = '''\
def add(a, b):
    return a + b

def greet(name, greeting="hi"):
    return f"{greeting} {name}"

def load(path):
    return open(path).read()

class Client:
    def send(self, msg):
        return msg

    def close(self):
        return None
'''
BASE_JS = '''\
export function add(a, b) { return a + b; }
export function greet(name, greeting) { return greeting + " " + name; }
'''

# (label, expected_breaking, file, new_content)
CASES = [
    ("py remove function", True, "m.py", BASE_PY.replace('def load(path):\n    return open(path).read()\n\n', '')),
    ("py rename function", True, "m.py", BASE_PY.replace("def load(", "def load_file(")),
    ("py add required param", True, "m.py", BASE_PY.replace("def add(a, b)", "def add(a, b, c)")),
    ("py remove param", True, "m.py", BASE_PY.replace("def add(a, b)", "def add(a)").replace("a + b", "a")),
    ("py remove method", True, "m.py", BASE_PY.replace("    def close(self):\n        return None\n", "")),
    ("py remove class", True, "m.py", BASE_PY.split("class Client")[0]),
    ("py add function", False, "m.py", BASE_PY + "\ndef sub(a, b):\n    return a - b\n"),
    ("py add optional param", False, "m.py", BASE_PY.replace("def add(a, b)", "def add(a, b, c=0)")),
    ("py docstring only", False, "m.py", BASE_PY.replace("def add(a, b):\n", 'def add(a, b):\n    """Add."""\n')),
    ("py body change same signature", False, "m.py", BASE_PY.replace("return a + b", "return b + a")),
    ("py comment only", False, "m.py", BASE_PY + "\n# trailing comment\n"),
    ("js remove export function", True, "m.js", BASE_JS.replace('export function greet(name, greeting) { return greeting + " " + name; }\n', '')),
    ("js change param count", True, "m.js", BASE_JS.replace("add(a, b)", "add(a, b, c)")),
    ("js add function", False, "m.js", BASE_JS + "export function sub(a, b) { return a - b; }\n"),
]


def planted(out):
    from src.git_analyzer import GitAnalyzer
    from src.breaking_change_detector import BreakingChangeDetector
    res = []
    for label, expected, fname, content in CASES:
        with tempfile.TemporaryDirectory() as d:
            run(d, "init", "-q")
            base = BASE_PY if fname.endswith(".py") else BASE_JS
            open(os.path.join(d, fname), "w").write(base)
            run(d, "add", "."); run(d, "commit", "-qm", "base")
            open(os.path.join(d, fname), "w").write(content)
            run(d, "commit", "-qam", "change")
            ga = GitAnalyzer(d)
            head = ga.repo.head.commit.hexsha
            det = BreakingChangeDetector(ga)
            found = det.analyze_commit_for_breaking_changes(head, ga)
            res.append(dict(case=label, expected_breaking=expected, n_flagged=len(found),
                            flagged=bool(found),
                            types=sorted({b.change_type.value for b in found}),
                            components=sorted({b.affected_component for b in found})[:6]))
    tp = sum(r["expected_breaking"] and r["flagged"] for r in res)
    fn = sum(r["expected_breaking"] and not r["flagged"] for r in res)
    fp = sum((not r["expected_breaking"]) and r["flagged"] for r in res)
    tn = sum((not r["expected_breaking"]) and not r["flagged"] for r in res)
    summary = dict(n_cases=len(res), TP=tp, FN=fn, FP=fp, TN=tn)
    json.dump(dict(summary=summary, cases=res), open(out, "w"), indent=1)
    print(json.dumps(summary)); [print(r) for r in res]


def history(repo_path, out):
    from src.git_analyzer import GitAnalyzer
    from src.breaking_change_detector import BreakingChangeDetector
    ga = GitAnalyzer(repo_path)
    det = BreakingChangeDetector(ga)
    rows = []
    for c in commits_oldest_first(ga.repo):
        if not c.parents:
            continue
        found = det.analyze_commit_for_breaking_changes(c.hexsha, ga)
        rows.append(dict(commit=c.hexsha[:7], author=c.author.name, msg=c.message.strip().split("\n")[0][:70],
                         n_breaking=len(found),
                         types=sorted({b.change_type.value for b in found}),
                         severities=sorted({b.severity.value for b in found}),
                         components=[b.affected_component for b in found][:5]))
    summary = dict(n_commits=len(rows), commits_with_findings=sum(r["n_breaking"] > 0 for r in rows),
                   total_findings=sum(r["n_breaking"] for r in rows))
    json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "drift":
        m = a[a.index("--model") + 1] if "--model" in a else "microsoft/codebert-base"
        drift(a[1], a[2], m)
    elif a[0] == "planted":
        planted(a[1])
    elif a[0] == "history":
        history(a[1], a[2])
