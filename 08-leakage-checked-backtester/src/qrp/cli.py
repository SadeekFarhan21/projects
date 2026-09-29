"""qrp command line: ingest data, inspect the store, run research configs, compare runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime


def cmd_ingest(a):
    from qrp.data import binance
    from qrp.data.store import BarStore
    print(f"listing USDT symbols on data.binance.vision ...", flush=True)
    syms = binance.list_usdt_symbols()
    print(f"{len(syms)} candidate symbols; selecting up to {a.max_symbols} by months of history",
          flush=True)
    sel = binance.select_symbols(syms, a.start, a.end, a.max_symbols)
    df, rep = binance.download_selected(sel)
    store = BarStore(a.root)
    iid = store.ingest(df, source="binance-spot-monthly-1d",
                       params={"start": a.start, "end": a.end, "max_symbols": a.max_symbols})
    print(json.dumps({"ingest_id": iid, **rep.__dict__}, indent=2))


def cmd_info(a):
    from qrp.data.store import BarStore
    store = BarStore(a.root)
    as_of = datetime.fromisoformat(a.as_of) if a.as_of else None
    print(store.ingests())
    syms = store.symbols(as_of)
    print(f"symbols visible as_of={a.as_of or 'now'}: {syms.height}  snapshot={store.snapshot_id(as_of)}")
    print(syms.head(10))


def cmd_run(a):
    from qrp.research import load_config, run_research
    cfg = load_config(a.config, a.set)
    out = run_research(cfg)
    print(f"run {out['run'].id}")
    print(json.dumps(out["metrics"], indent=2))


def cmd_audit(a):
    from qrp.features.spec import FeatureSpec, audit_causality
    from qrp.research import load_config, load_panel
    cfg = load_config(a.config, a.set)
    panel, _ = load_panel(cfg)
    specs = [FeatureSpec.from_dict(f) for f in cfg["features"]]
    bad = 0
    for r in audit_causality(panel, specs):
        status = "ok" if r.causal else f"LEAK row {r.first_bad_row} cut {r.cut} ({r.method})"
        bad += not r.causal
        print(f"{r.feature:24s} {status}")
    sys.exit(1 if bad else 0)


def cmd_runs(a):
    from qrp import tracker
    if a.action == "list":
        runs = tracker.load_runs(a.root)
        if a.sort:
            runs.sort(key=lambda r: r["metrics"].get(a.sort, float("-inf")) or float("-inf"),
                      reverse=True)
        print(tracker.list_table(runs))
    elif a.action == "show":
        r = tracker.find_run(a.root, a.ids[0])
        print(json.dumps({k: r[k] for k in ("id", "meta", "metrics", "config")}, indent=2))
    elif a.action == "compare":
        print(tracker.compare_table([tracker.find_run(a.root, k) for k in a.ids]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="qrp", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="download daily Binance spot bars into the store")
    s.add_argument("--root", default="data")
    s.add_argument("--start", default="2022-01")
    s.add_argument("--end", default="2026-08")
    s.add_argument("--max-symbols", type=int, default=300)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("info", help="show ingests and symbols as of a time")
    s.add_argument("--root", default="data")
    s.add_argument("--as-of", default=None)
    s.set_defaults(fn=cmd_info)

    for name, fn, hlp in (("run", cmd_run, "run a research config and track it"),
                          ("audit", cmd_audit, "leakage-audit a config's features")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("config")
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. costs.fee_bps=0")
        s.set_defaults(fn=fn)

    s = sub.add_parser("runs", help="list, show or compare tracked runs")
    s.add_argument("action", choices=["list", "show", "compare"])
    s.add_argument("ids", nargs="*")
    s.add_argument("--root", default="runs")
    s.add_argument("--sort", default=None, help="metric to sort by (list only)")
    s.set_defaults(fn=cmd_runs)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
