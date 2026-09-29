"""Benchmark of Aegis first-stage gate (AnomalyDetectionEngine) on a synthetic stream that
mirrors backend/streaming/producer.py distributions (8% fraud / 12% elevated / 80% normal).
Domestic merchants are MY stand-ins (Nessie merchants need an API key)."""
import sys, random, time, json, math
from datetime import datetime, timedelta
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from backend.anomaly_detection.engine import AnomalyDetectionEngine
from backend.streaming import producer as P

DOMESTIC = [("Coffee",40.44,-79.99),("Grocery",40.45,-79.95),("Gas",40.43,-80.0),("Restaurant",40.44,-79.98),
 ("Retail",40.46,-79.92),("Transport",40.44,-79.99),("Online",40.44,-79.99),("Subscription",40.44,-79.99),
 ("Pharmacy",40.45,-79.97),("Entertainment",40.44,-79.96),("Electronics",40.47,-79.93),("Clothing",40.45,-79.94),
 ("Health",40.43,-79.98),("Lodging",40.44,-79.99),("Automotive",40.42,-79.95)]

def gen(n, seed, start):
    random.seed(seed)
    t = start; out = []
    for i in range(n):
        t += timedelta(seconds=random.uniform(1, 3))
        roll = random.random()
        acc = random.choice(P.ACCOUNT_IDS)
        if roll < 0.08:
            m = random.choice(P.ANOMALY_MERCHANTS); amt = P._get_anomaly_amount()
            d = dict(category=m["category"], latitude=m["lat"], longitude=m["lng"], city=m["city"], country=m["country"]); label = "fraud"
        else:
            cat, la, lo = random.choice(DOMESTIC); amt = P._get_amount_for_category(cat)
            label = "normal"
            if roll < 0.20:
                amt = round(amt * random.uniform(2.0, 5.0), 2); label = "elevated"
            d = dict(category=cat, latitude=la+random.uniform(-.01,.01), longitude=lo+random.uniform(-.01,.01), city="Pittsburgh", country="US")
        out.append((dict(id=str(i), account_id=acc, amount=amt, timestamp=t.isoformat(), **d), label))
    return out

def run(seed, start, n=5000):
    np.random.seed(seed)
    eng = AnomalyDetectionEngine()
    data = gen(n, seed, start)
    flags = []; ifonly = []
    t0 = time.perf_counter()
    tot_detect = 0.0
    for txn, lab in data:
        f = eng._extract_features(txn)
        ifonly.append(bool(-eng.model.decision_function(f)[0]*2 > 0.55))
        a = time.perf_counter(); r = eng.detect(txn); tot_detect += time.perf_counter()-a
        flags.append(r.is_anomaly)
    labs = [l for _, l in data]
    def stats(fl):
        tp = sum(f and l=="fraud" for f,l in zip(fl,labs)); fp = sum(f and l!="fraud" for f,l in zip(fl,labs))
        fn = sum((not f) and l=="fraud" for f,l in zip(fl,labs)); nf=sum(l=="fraud" for l in labs)
        fpn = sum(f and l=="normal" for f,l in zip(fl,labs)); fpe = sum(f and l=="elevated" for f,l in zip(fl,labs))
        return dict(precision=tp/max(tp+fp,1), recall=tp/nf, flagged_frac=sum(fl)/len(fl),
                    fp_normal=fpn, fp_elevated=fpe, n_normal=labs.count("normal"), n_elev=labs.count("elevated"), n_fraud=nf,
                    flagged=sum(fl))
    return dict(seed=seed, start=str(start), gate=stats(flags), iforest_only=stats(ifonly), detect_per_sec=n/tot_detect)

if __name__ == "__main__":
    res = []
    for start in [datetime(2026,2,7,12,0), datetime(2026,2,7,3,0)]:
        for seed in range(3):
            res.append(run(seed, start)); print('done',start,seed,flush=True); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','gate_bench.json'),'w'), indent=1)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','gate_bench.json'),'w'), indent=1)
    for r in res: print(r["start"], r["seed"], {k: round(v,3) if isinstance(v,float) else v for k,v in r["gate"].items() if k in("precision","recall","flagged_frac")}, "IFonly", {k: round(v,3) for k,v in r["iforest_only"].items() if k in("precision","recall","flagged_frac")}, round(r["detect_per_sec"]))
