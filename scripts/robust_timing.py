#!/usr/bin/env python3
"""Robust microbenchmark (reviewer #3/#4): inner-loop batched timing with
median and IQR over many blocks, GC disabled. Distinguishes C-vs-Python-loop
from C-vs-vectorised-NumPy. Run against v9.0.0."""
import gc, json, os, statistics, sys, time
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from metbit._native import vip_scores as c_vip, pearson_columns as c_pear, column_variances as c_var

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/hpc900"

def timed(fn, inner, blocks=25, warm=3):
    gc.disable()
    for _ in range(warm):
        for _ in range(inner): fn()
    per=[]
    for _ in range(blocks):
        t=time.perf_counter()
        for _ in range(inner): fn()
        per.append((time.perf_counter()-t)*1e3/inner)  # ms per call
    gc.enable()
    per.sort()
    q1=per[len(per)//4]; q3=per[(3*len(per))//4]
    return {"min_ms":round(min(per),6),"median_ms":round(statistics.median(per),6),
            "iqr_ms":round(q3-q1,6),"blocks":blocks,"inner":inner}

def pls_mats(n,p,h=3,seed=0):
    rng=np.random.default_rng(seed); X=rng.standard_normal((n,p)); y=rng.integers(0,2,n).astype(float)
    m=PLSRegression(n_components=min(h,min(n,p)-1)).fit(X,y)
    return m.x_scores_.astype(float), m.x_weights_.astype(float), m.y_loadings_.astype(float)

def numpy_vip(t,w,q,p):
    S=np.einsum("ij,ij->j",t,t)*(q.ravel()**2); nz=np.linalg.norm(w,axis=0); nz[nz==0]=1
    return np.sqrt(p*((w/nz)**2@S)/S.sum())
def vip_loop(t,w,q):
    p,h=w.shape; s=np.diag(t.T@t@q.T@q).reshape(h,-1); tot=float(np.sum(s)); out=np.zeros(p)
    for i in range(p):
        wt=np.array([(w[i,j]/np.linalg.norm(w[:,j]))**2 for j in range(h)])
        out[i]=float(np.sqrt(p*(s.T@wt)/tot).squeeze())
    return out

res={"vip":[], "pearson":[], "variance":[]}
def dump():
    os.makedirs(OUT,exist_ok=True)
    json.dump({"method":"inner-loop batched, GC disabled, min/median/IQR over 25 blocks (10 for python_loop)",
               "results":res}, open(os.path.join(OUT,"ROBUST_TIMING.json"),"w"), indent=2)
# choose inner counts so each block is a few ms
for p,inner in [(1000,2000),(5000,1000),(20000,300)]:
    t,w,q=pls_mats(80,p)
    row={"n":80,"p":p,
         "C":timed(lambda:c_vip(t,w,q), inner),
         "numpy":timed(lambda:numpy_vip(t,w,q,p), inner)}
    if p<=5000:
        row["python_loop"]=timed(lambda:vip_loop(t,w,q), max(3,inner//200), blocks=10)
        row["C_vs_numpy"]=round(row["numpy"]["median_ms"]/row["C"]["median_ms"],2)
        row["C_vs_loop"]=round(row["python_loop"]["median_ms"]/row["C"]["median_ms"],1)
    res["vip"].append(row); print("vip",p,"C_vs_numpy=",row.get("C_vs_numpy"),flush=True); dump()

for n,p,inner in [(200,10000,200),(500,30000,60)]:
    mat=np.random.default_rng(0).standard_normal((n,p)); a=p//2
    def npc():
        ac=mat[:,a]-mat[:,a].mean(); C=mat-mat.mean(axis=0)
        d=np.sqrt(np.dot(ac,ac)*np.einsum("ij,ij->j",C,C)); return np.clip(ac@C/d,-1,1)
    row={"n":n,"p":p,"C":timed(lambda:c_pear(mat,anchor_index=a),inner),
         "numpy_fullcopy":timed(npc,inner)}
    row["C_vs_numpy"]=round(row["numpy_fullcopy"]["median_ms"]/row["C"]["median_ms"],2)
    res["pearson"].append(row); print("pearson",n,p,"C_vs_numpy=",row["C_vs_numpy"],flush=True); dump()

for n,p,inner in [(200,50000,20),(500,100000,6)]:
    mat=np.random.default_rng(0).standard_normal((n,p))
    row={"n":n,"p":p,"C":timed(lambda:c_var(mat),inner),
         "numpy":timed(lambda:mat.var(axis=0,ddof=1),inner)}
    row["C_vs_numpy"]=round(row["numpy"]["median_ms"]/row["C"]["median_ms"],2)
    res["variance"].append(row); print("variance",n,p,"C_vs_numpy=",row["C_vs_numpy"],flush=True); dump()

os.makedirs(OUT,exist_ok=True)
json.dump({"method":"inner-loop batched, GC disabled, min/median/IQR over 25 blocks (10 for python_loop)",
           "results":res}, open(os.path.join(OUT,"ROBUST_TIMING.json"),"w"), indent=2)
print("wrote", os.path.join(OUT,"ROBUST_TIMING.json"))
