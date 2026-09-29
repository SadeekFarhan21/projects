"""Recompute cure_seq - cure deltas from the committed Figure-6 results.json files (Jeffrey Xie's runs).
Run from project root: python results/figure6_deltas.py  (stdlib only; writes results/figure6_deltas.json)"""
import json, os
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
out = {}
for name in ("figure6_results", "figure6_results_100"):
    r = json.load(open(f"{root}/evaluation/{name}/results.json"))["results"]
    c = {x["n_erased"]: x for x in r["cure"]["checkpoints"]}
    s = {x["n_erased"]: x for x in r["cure_seq"]["checkpoints"]}
    rows = []
    for k in sorted(c):
        row = {"k": k}
        for m in ("lpips_e_mean", "lpips_u_mean", "clip_u_mean"):
            row[f"cure_{m}"] = round(c[k][m], 4); row[f"seq_{m}"] = round(s[k][m], 4)
            row[f"delta_{m}"] = round(s[k][m] - c[k][m], 4)
        row["cure_lpips_u_std"] = round(c[k]["lpips_u_std"], 4); row["seq_lpips_u_std"] = round(s[k]["lpips_u_std"], 4)
        rows.append(row)
    out[name] = rows
    print(name)
    for r_ in rows: print(" k=%-3d dLPIPSu=%+.4f dCLIPu=%+.3f dLPIPSe=%+.4f (std_u cure %.3f seq %.3f)" % (r_["k"], r_["delta_lpips_u_mean"], r_["delta_clip_u_mean"], r_["delta_lpips_e_mean"], r_["cure_lpips_u_std"], r_["seq_lpips_u_std"]))
json.dump(out, open(f"{here}/figure6_deltas.json", "w"), indent=2)
