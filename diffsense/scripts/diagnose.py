"""Two diagnostics behind the drift numbers (see README):
 (a) pairwise cosine of raw CodeBERT mean-pooled vectors vs MiniLM vectors for unrelated code snippets
 (b) diff direction: GitAnalyzer.extract_diff_info uses commit.diff(parent); check which side lands in added_lines
"""
import os, sys, json, tempfile, subprocess, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend")); logging.disable(logging.CRITICAL)
from src.embedding_engine import CodeEmbedder, TextEmbedder
from src.git_analyzer import GitAnalyzer
from sklearn.metrics.pairwise import cosine_similarity as cs
snips = ["def add(a, b):\n    return a + b", "SELECT name FROM users WHERE id = 3;",
         "class Foo { public static void main(String[] a) { System.out.println(1); } }",
         "for (int i = 0; i < n; i++) sum += x[i];", "import numpy as np\nx = np.zeros((3, 3))"]
out = {}
for name, emb in [("codebert", CodeEmbedder("microsoft/codebert-base")), ("minilm_as_code", CodeEmbedder("nonexistent/none"))]:
    v = [emb.embed_code(s) for s in snips]
    sims = [float(cs([v[i]], [v[j]])[0][0]) for i in range(len(v)) for j in range(i + 1, len(v))]
    out[name] = dict(pairs=len(sims), min_cosine=round(min(sims), 4), max_cosine=round(max(sims), 4))
d = tempfile.mkdtemp(); g = lambda *a: subprocess.run(["git", *a], cwd=d, check=True, capture_output=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
g("init", "-q"); open(d + "/a.py", "w").write("OLDLINE = 1\n"); g("add", "."); g("commit", "-qm", "1")
open(d + "/a.py", "w").write("NEWLINE = 2\n"); g("commit", "-qam", "2")
ga = GitAnalyzer(d); info = ga.extract_diff_info(ga.repo.head.commit)[0]
out["diff_direction"] = dict(added_lines=info.added_lines, removed_lines=info.removed_lines)
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "diagnostics.json"), "w"), indent=1)
