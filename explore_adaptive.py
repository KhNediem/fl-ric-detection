"""Exploratory (Discussion section): channel-aware features (CA) and online recalibration (OR) of D1,
on synthetic and trace-driven channels, 12 runs. usage: python3 explore_adaptive.py"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from core import CFG, SCENARIOS, generate, features, Detectors
from unified import *
import trace, adaptive as ad
class NoAE(Detectors):
    def _fit_ae(self,*a,**k): pass
m=100; T=400; ev0=200
def gen(scn,s):
    if scn in ('gauss','ar'):
        rng=np.random.default_rng([s,SCENARIOS.index(scn),m]); par=generate(scn,T,rng)
    else:
        rng=np.random.default_rng([s,10+['static','driving'].index(scn),m]); par=trace.generate_trace(scn,T,rng)
    return par, rng
def evaluate(F_fn, online, X, par, obs, byzset):
    F=F_fn(X,obs)
    if online: fl,_=ad.d1_online(F,m,ev0)
    else:
        S=ad.d1_static(F,m); fl=flags(S,thr(S,m,0.01))
    return fl
rows=[]
for scn in ['gauss','ar','static','driving']:
  for s in range(12):
    par,rng=gen(scn,s); raw=features(par); perm=rng.permutation(N)
    obs=par['logc']+ad.SIG_OBS*np.random.default_rng([s,555]).standard_normal(par['logc'].shape)
    byz=np.zeros(N,bool); byz[perm[:10]]=True; allu=np.ones(N,bool)
    det=NoAE(raw,m,s)
    XA1=apply_attack('A1',3,par,raw,det,byz,ev0); XA8=apply_attack('A8',0.5,par,raw,det,byz,ev0)
    b6=np.zeros(N,bool); b6[perm[:15]]=True; lc=par['logc'].copy(); lc[:,ev0:]-=np.log(1.3); X6=features(par,lc)
    obs6=obs.copy()  # contention does not change the reported channel quality
    for name,Ffn in [('plain',lambda X,o: np.log1p(X).astype(float)),('CA',lambda X,o: ad.ca_features(X,o,m))]:
      for online in (False,True):
        v=name+('+OR' if online else '')
        fl=evaluate(Ffn,online,raw,par,obs,None)
        r=dict(scn=scn,seed=s,var=v,fpr=fl[:,ev0:].mean(),fi=iso_stats(decide(fl,3,5),ev0,allu)[0])
        fl=evaluate(Ffn,online,XA1,par,obs,None); r['A1_iso']=iso_stats(decide(fl,3,5),ev0,byz)[0]
        fl=evaluate(Ffn,online,XA8,par,obs,None); r['A8_iso']=iso_stats(decide(fl,3,5),ev0,byz)[0]
        fl=evaluate(Ffn,online,X6,par,obs6,None); r['A6_fi']=iso_stats(decide(fl,3,5),ev0,~b6)[0]
        rows.append(r)
d=pd.DataFrame(rows); g=d.groupby(['scn','var'])[['fpr','fi','A1_iso','A8_iso','A6_fi']].mean()*100
print(g.round(1).to_string())
