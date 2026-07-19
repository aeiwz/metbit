#!/usr/bin/env python3
"""Reviewer-requested benchmarks (run against v9.0.0):
   A) process-level peak RSS (subprocess isolation) for baseline vs C kernels,
      distinguishing interpreter+input floor from the avoided temporary copy;
   B) equivalent compute-only STOCSY: standard correlation vs ChunkedSTOCSY.compute().
Writes reports/hpc900/REVIEWER_BENCH.json .
"""
import json, os, subprocess, sys, time, statistics, resource, platform

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/hpc900"
PY = sys.executable
KB = 1 if platform.system() == "Linux" else (1/1024.0)  # ru_maxrss: KB on Linux, bytes on mac

# ---- A) process-level peak RSS via isolated child processes -----------------
CHILD = r'''
import numpy as np, resource, sys
mode, n, p = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# standard_normal already returns float64 C-contiguous; avoid an astype copy that
# would itself set peak RSS and mask the kernel's own temporary allocation.
mat = np.random.default_rng(0).standard_normal((n,p))
anchor = p//2
if mode == "input_only":
    pass
elif mode == "pearson_baseline":
    a=mat[:,anchor]; ac=a-a.mean(); C=mat-mat.mean(axis=0)
    num=ac@C; den=np.sqrt(np.dot(ac,ac)*np.einsum("ij,ij->j",C,C))
    r=np.clip(num/den,-1,1)
elif mode == "pearson_C":
    from metbit._native import pearson_columns
    r=pearson_columns(mat, anchor_index=anchor)
elif mode == "var_baseline":
    v=mat.var(axis=0, ddof=1)
elif mode == "var_C":
    from metbit._native import column_variances
    v=column_variances(mat)
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak_kb)
'''
open("/tmp/_rss_child.py","w").write(CHILD)

def child_rss(mode, n, p, reps=3):
    vals=[]
    for _ in range(reps):
        out=subprocess.check_output([PY,"/tmp/_rss_child.py",mode,str(n),str(p)]).decode().strip()
        vals.append(int(out.splitlines()[-1]) * KB / 1024.0)  # -> MB
    return min(vals)

rss_cases=[
    ("pearson", 500, 30000, ["input_only","pearson_baseline","pearson_C"]),
    ("variance", 500, 100000, ["input_only","var_baseline","var_C"]),
]
rss_results=[]
for name,n,p,modes in rss_cases:
    input_mb = n*p*8/1e6
    row={"kernel":name,"n":n,"p":p,"input_matrix_mb":round(input_mb,1),"rss_mb":{}}
    for m in modes:
        row["rss_mb"][m]=round(child_rss(m,n,p),1)
    rss_results.append(row)
    print("RSS", name, n, "x", p, row["rss_mb"], flush=True)

# ---- B) equivalent compute-only STOCSY --------------------------------------
import numpy as np, pandas as pd
from metbit._native import pearson_columns
from metbit import ChunkedSTOCSY
def bench(fn, reps=7, warm=1):
    for _ in range(warm): fn()
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    return ts
stocsy_results=[]
for n,p in [(50,2000),(100,5000)]:
    ppm=np.linspace(9.5,0.5,p)
    spectra=pd.DataFrame(np.random.default_rng(0).standard_normal((n,p)), columns=ppm.tolist())
    mat=np.ascontiguousarray(spectra.to_numpy(dtype=np.float64)); idx=p//3
    std=bench(lambda: pearson_columns(mat, anchor_index=idx))          # compute-only standard corr
    chk=bench(lambda: ChunkedSTOCSY(chunk_size=5000).compute(spectra, anchor_ppm_value=float(ppm[idx])))
    # numerical agreement
    r_std=pearson_columns(mat, anchor_index=idx)
    try:
        r_chk=ChunkedSTOCSY(chunk_size=5000).compute(spectra, anchor_ppm_value=float(ppm[idx]))
        r_chk=np.asarray(r_chk[0] if isinstance(r_chk,tuple) else r_chk).ravel()
        maxdiff=float(np.nanmax(np.abs(r_std - r_chk[:len(r_std)])))
    except Exception as e:
        maxdiff=None
    row={"n":n,"p":p,"std_corr_min_ms":round(min(std),4),"chunked_compute_min_ms":round(min(chk),4),
         "ratio_std_over_chunked":round(min(std)/min(chk),3),
         "std_corr_median_ms":round(statistics.median(std),4),
         "chunked_median_ms":round(statistics.median(chk),4),
         "max_abs_diff":maxdiff}
    stocsy_results.append(row); print("STOCSY compute-only", row, flush=True)

os.makedirs(OUT, exist_ok=True)
json.dump({"platform":platform.platform(),"python":platform.python_version(),
           "process_rss":rss_results,"stocsy_compute_only":stocsy_results},
          open(os.path.join(OUT,"REVIEWER_BENCH.json"),"w"), indent=2)
print("wrote", os.path.join(OUT,"REVIEWER_BENCH.json"))
