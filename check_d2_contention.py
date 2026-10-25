import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from core import CFG, SCENARIOS, generate, features, Detectors
from unified import *
res={}
for scn in ['gauss','ar']:
    out=[]
    for seed in range(4):
        m=100; rng=np.random.default_rng([seed,SCENARIOS.index(scn),m]); T=m+300; ev0=m+100
        par=generate(scn,T,rng); raw=features(par); det=Detectors(raw,m,seed)
        S0=det.scores(raw); perm=rng.permutation(N)
        b6=np.zeros(N,bool); b6[perm[:15]]=True
        lc=par['logc'].copy(); lc[:,ev0:]-=np.log(1.3); X=features(par,lc); S=det.scores(X)
        f=np.log1p(raw); z=(f-det.mu[:,None,:])/det.sd_f[:,None,:]
        fX=np.log1p(X); zX=(fX-det.mu[:,None,:])/det.sd_f[:,None,:]
        for k in ['D2','D1lw']:
            th=thr(S[k],m,0.01)
            cal=S0[k][:,m:m+100]; evb=S0[k][~b6][:,ev0:]; evl=S[k][~b6][:,ev0:]
            out.append((k,th,np.median(cal),np.median(evl), (evl>th).mean(),
                        np.mean(zX[~b6][:,ev0:,2]-z[~b6][:,ev0:,2]), np.mean(zX[~b6][:,ev0:,1]-z[~b6][:,ev0:,1]),
                        np.mean(z[~b6][:,ev0:,2]), np.mean(z[~b6][:,ev0:,1])))
    for k in ['D2','D1lw']:
        a=np.array([o[1:] for o in out if o[0]==k]).mean(0)
        print(scn,k,"theta %.3f  med_cal %.3f  med_eval_loaded %.3f  flagrate %.3f  dz_d %.2f dz_tau %.2f  z_d(no load,eval) %.2f z_tau %.2f"%tuple(a))
print("--- constant offset windows")
for scn in ['gauss','ar']:
    r=[]
    for seed in range(4):
        m=100; rng=np.random.default_rng([seed,SCENARIOS.index(scn),m]); T=m+300
        par=generate(scn,T,rng); raw=features(par); det=Detectors(raw,m,seed)
        w0=np.zeros((1,15)); w1=np.tile([0,0.66,1.05],5)[None,:]
        e0=((det.ae.predict(w0)-w0)**2).mean(); e1=((det.ae.predict(w1)-w1)**2).mean()
        r.append((e0,e1))
    print(scn, np.mean(r,0).round(3))
