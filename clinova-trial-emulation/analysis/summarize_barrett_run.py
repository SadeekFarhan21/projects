"""Offline summary of the committed Barrett-2006 pipeline run artifacts.
Reads only files in backend/run/benchmark-barrett-*; calls no APIs. Writes results/barrett_run_summary.json."""
import csv, glob, json, os, re, collections
from datetime import datetime
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run = glob.glob(os.path.join(root, "backend/run/benchmark-barrett-*"))[0]
out = {}
# timings from run_log
log = [l.strip() for l in open(f"{run}/run_log.txt") if l.strip()]
ts = [(datetime.strptime(l[1:20], "%Y-%m-%d %H:%M:%S"), l[22:]) for l in log]
out["total_seconds"] = (ts[-1][0] - ts[0][0]).total_seconds()
steps = {}
for (t0, m0), (t1, m1) in zip(ts, ts[1:]):
    steps[m0] = (t1 - t0).total_seconds()
out["seconds_until_next_log_line"] = steps
out["log_chars"] = {m: re.search(r"(\d+) chars", m).group(1) for _, m in ts if "chars" in m}
# validator iterations
txt = open(f"{run}/step2b_validation_feedback.txt").read()
iters = []
for blk in re.split(r"=== Iteration (\d+) ===", txt)[1:]:
    pass
parts = re.split(r"=== Iteration (\d+) ===", txt)
for n, body in zip(parts[1::2], parts[2::2]):
    j = json.loads(body.strip())
    iters.append({"iteration": int(n), "status": j["status"], "failed_gates": j["failed_gates"],
                  "issues": [(i["gate"], i["severity"]) for i in j["issues"]],
                  "issue_counts": dict(collections.Counter(f'{i["gate"]}:{i["severity"]}' for i in j["issues"])),
                  "n_passed_gates": len(j["passed_gates"])})
out["validator_iterations"] = iters
# spec sizes
out["spec_lines"] = {os.path.basename(p): sum(1 for _ in open(p)) for p in sorted(glob.glob(f"{run}/step2a_design_spec*.md"))}
out["step3_code_lines"] = sum(1 for _ in open(f"{run}/step3_code.py"))
out["lookup_terms_file_lines"] = sum(1 for l in open(f"{run}/step2c_lookup_terms.txt") if l.strip())
# OMOP CSV
rows = list(csv.DictReader(open(f"{run}/step2c_omop_results.csv")))
out["omop_csv_rows"] = len(rows)
out["omop_csv_distinct_query_terms"] = len({r["query_term"] for r in rows})
out["match_type_counts_all_rows"] = dict(collections.Counter(r["match_type"] for r in rows))
top1 = [r for r in rows if r["rank"] == "1"]
out["top1_rows"] = len(top1)
out["top1_match_type_counts"] = dict(collections.Counter(r["match_type"] for r in top1))
out["top1_confidence_below_0.5"] = [(r["query_term"], r["concept_name"], r["vocabulary_id"], r["confidence"]) for r in top1 if float(r["confidence"]) < 0.5]
out["top1_missing_analytics_concept_id"] = [r["query_term"] for r in top1 if not r["analytics_concept_id"]]
out["top1_confidence_ge_0.95"] = sum(float(r["confidence"]) >= 0.95 for r in top1)
out["top1_by_domain"] = dict(collections.Counter(r["domain_id"] for r in top1))
terms = [l.split("|")[0] for l in open(f"{run}/step2c_lookup_terms.txt") if l.strip()]
seen = list(dict.fromkeys(r["query_term"] for r in rows))
out["terms_in_lookup_file"] = len(terms)
out["csv_terms_are_first_n_of_lookup_file"] = seen == terms[:len(seen)]
os.makedirs(os.path.join(root, "results"), exist_ok=True)
json.dump(out, open(os.path.join(root, "results/barrett_run_summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
