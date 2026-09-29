import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys; sys.argv=['x']
from gate_bench import *
np.random.seed(0); eng=AnomalyDetectionEngine(); data=gen(3000,0,datetime(2026,2,7,12,0))
rows=[]
for txn,lab in data:
    f=eng._extract_features(txn); raw=float(eng.model.decision_function(f)[0]); r=eng.detect(txn)
    rows.append((lab,raw,r.score,r.is_anomaly,txn['amount'],txn['country']))
for lab in ['normal','elevated','fraud']:
    rr=[x for x in rows if x[0]==lab]; raws=np.array([x[1] for x in rr]); sc=np.array([x[2] for x in rr])
    print(lab,len(rr),'raw IF min/med/max',raws.min().round(3),np.median(raws).round(3),raws.max().round(3),'gate score med',np.median(sc).round(3),'flag rate',np.mean([x[3] for x in rr]).round(3))
fr=[x for x in rows if x[0]=='fraud']
for lo,hi in [(0,500),(500,1000),(1000,5000),(5000,1e9)]:
    b=[x for x in fr if lo<=x[4]<hi]; print('fraud amount',lo,hi,len(b),'recall',np.mean([x[3] for x in b]).round(3) if b else None)
print('fraud amount>1000 share', np.mean([x[4]>1000 for x in fr]).round(3))
