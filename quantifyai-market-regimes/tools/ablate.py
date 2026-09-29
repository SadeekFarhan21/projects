# Post-hoc audit written after the fact (not by the original team).
# Usage from the repo root, after main.py: python tools/ablate.py <output_dir>
import sys; OUT = sys.argv[1] if len(sys.argv) > 1 else 'out1'
import sys, logging, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.')
logging.disable(logging.CRITICAL)
import pandas as pd, numpy as np
from src.enhanced_backtester import EnhancedBacktester
df0 = pd.read_csv(OUT+'/market_predictions.csv', index_col=0, parse_dates=True)
def run(df, name='combined_anomaly_regime'):
    bt = EnhancedBacktester(df.copy(), use_optimized=True)
    bt.run_combined_anomaly_regime_strategy()
    m = bt.calculate_metrics('combined_anomaly_regime')
    return m
def fmt(m): return f"TR={m['Total_Return']:.2f}% Ann={m['Annualized_Return']*100:.2f}% Sharpe={m['Sharpe_Ratio']:.2f} MDD={m['Max_Drawdown']:.2f}%"
print("baseline (as published)", fmt(run(df0)))
d=df0.copy(); d['ensemble_anomaly']=1; print("A no anomalies", fmt(run(d)))
# causal anomaly: |252d rolling z of return| > 3
r=d['SP500'].pct_change(); z=(r-r.rolling(252,min_periods=60).mean())/r.rolling(252,min_periods=60).std()
d=df0.copy(); d['ensemble_anomaly']=np.where(z.abs()>3,-1,1); print("B causal rolling-z anomalies n=",(d.ensemble_anomaly==-1).sum(), fmt(run(d)))
# placebo: same number of anomaly flags at random dates
n=(df0.ensemble_anomaly==-1).sum(); res=[]
for seed in range(200):
    rng=np.random.default_rng(seed); idx=rng.choice(len(df0),n,replace=False)
    a=np.ones(len(df0)); a[idx]=-1; d=df0.copy(); d['ensemble_anomaly']=a; res.append(run(d))
tr=np.array([m['Total_Return'] for m in res]); sh=np.array([m['Sharpe_Ratio'] for m in res]); dd=np.array([m['Max_Drawdown'] for m in res])
print(f"C placebo random {n} anomaly days x200: TR mean {tr.mean():.2f} [p5 {np.percentile(tr,5):.2f}, p95 {np.percentile(tr,95):.2f}] Sharpe mean {sh.mean():.2f} [p5 {np.percentile(sh,5):.2f}, p95 {np.percentile(sh,95):.2f}] MDD mean {dd.mean():.2f} [worst {dd.min():.2f}, best {dd.max():.2f}]")
print("   share of placebo runs with Sharpe >= published:", (sh>=1.10).mean(), " TR >= published:", (tr>=56.68).mean())
# no predictions: always Bull
d=df0.copy(); d['Predicted_Market']='Bull'; print("D always-Bull + real anomalies", fmt(run(d)))
d['ensemble_anomaly']=1; print("E always-Bull + no anomalies (regime overlay only)", fmt(run(d)))
# persistence: predicted = causal current state
d=df0.copy(); d['Predicted_Market']=d['Market_State']; print("F predicted=current Market_State", fmt(run(d)))
# accuracy of Predicted_Market in the test period
ms=df0['Market_State']; pm=df0['Predicted_Market']
print("test-period acc vs same-day state", (pm==ms).mean(), " vs state 63d ahead", (pm[:-63]==ms.shift(-63)[:-63]).mean())
print("test-period pred dist", pm.value_counts().to_dict(), "state dist", ms.value_counts().to_dict())
print("persistence acc (state_t == state_t+63) test", (ms[:-63]==ms.shift(-63)[:-63]).mean())
# simple baselines on full-history price
px=df0['SP500']; ret=px.pct_change().fillna(0)
import math
bond=(1+df0['BondRate']/100)**(1/252)-1
def stat(pr,name):
    pv=(1+pr).cumprod(); tr=(pv.iloc[-1]-1)*100; mdd=(pv/pv.cummax()-1).min()*100
    print(f"{name}: TR={tr:.2f}% Sharpe={pr.mean()/pr.std()*math.sqrt(252):.2f} MDD={mdd:.2f}%")
stat(ret,"buy&hold")
for w in [0.5,0.6]: stat((w*ret+(1-w)*bond),f"constant {int(w*100)}/{int((1-w)*100)}")
sp_full=pd.read_csv(OUT+'/market_states.csv',index_col=0,parse_dates=True)
