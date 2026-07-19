"""Build a self-contained HTML report from the MTBLS1 benchmark artifacts in
result/. Regenerates metbit-result plots (spectra QC, PCA scores, VIP,
per-stage performance) and embeds them + the JSON reports into one HTML file.

Usage:
    python Benchmark/MTBLS1/build_report.py
Run benchmark_mtbls1.py first to populate result/.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULT = Path(__file__).resolve().parent / "result"


def png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def file_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def pca_plot():
    df = pd.read_csv(RESULT / "pca_scores.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for grp, sub in df.groupby("Group"):
        ax.scatter(sub["PC1"], sub["PC2"], label=grp, s=28, alpha=0.75, edgecolor="w", lw=0.4)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA scores (pareto scaling)")
    ax.legend(fontsize=8)
    return png_b64(fig)


def vip_plot(top_n=25):
    df = pd.read_csv(RESULT / "vip_scores.csv").sort_values("VIP", ascending=False).head(top_n)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh([f"{f:.3f}" for f in df["Features"]], df["VIP"], color="#c0392b")
    ax.set_xlabel("VIP score"); ax.set_ylabel("Feature (ppm)")
    ax.set_title(f"Top {top_n} OPLS-DA VIP features")
    ax.tick_params(axis="y", labelsize=7)
    return png_b64(fig)


def perf_plot(perf):
    stages = [s["stage"].split(" (")[0] for s in perf["stages"]]
    secs = [s["seconds"] for s in perf["stages"]]
    mem = [s["peak_mb"] for s in perf["stages"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.barh(stages[::-1], secs[::-1], color="#2980b9")
    a1.set_xlabel("seconds"); a1.set_title("Wall-clock per stage")
    a2.barh(stages[::-1], mem[::-1], color="#27ae60")
    a2.set_xlabel("peak MB"); a2.set_title("Peak memory per stage")
    for a in (a1, a2):
        a.tick_params(axis="y", labelsize=8)
    return png_b64(fig)


def main():
    perf = json.loads((RESULT / "performance_report.json").read_text())
    repro = json.loads((RESULT / "reproducibility_report.json").read_text())
    usab = json.loads((RESULT / "usability_report.json").read_text())
    opls = json.loads((RESULT / "opls_metrics.json").read_text())
    outlier = json.loads((RESULT / "outlier_report.json").read_text())
    env = perf["environment"]

    spectra_png = file_b64(RESULT / "spectra_qc.png")
    pca_png = pca_plot()
    vip_png = vip_plot()
    perf_png = perf_plot(perf)

    perf_rows = "".join(
        f"<tr><td>{s['stage']}</td><td>{s['seconds']}</td><td>{s['peak_mb']}</td></tr>"
        for s in perf["stages"]
    )
    env_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in env.items()
                       if not isinstance(v, dict))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>metbit MTBLS1 Benchmark Report</title>
<style>
 :root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--card:#fafafa;--accent:#c0392b}}
 @media(prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2c2f36;--card:#1e2127}}}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
 .wrap{{max-width:960px;margin:0 auto;padding:32px 20px 64px}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}}
 .sub{{color:var(--mut);margin:0 0 8px}}
 .kpis{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}}
 .kpi{{flex:1;min-width:130px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
 .kpi .v{{font-size:22px;font-weight:600}} .kpi .l{{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
 table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}} th{{color:var(--mut);font-weight:600}}
 figure{{margin:14px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}
 img{{max-width:100%;height:auto;display:block;margin:0 auto}} figcaption{{color:var(--mut);font-size:13px;margin-top:8px;text-align:center}}
 .note{{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;color:var(--mut);font-size:13px}}
 code{{font-family:ui-monospace,Menlo,monospace;font-size:13px}}
</style></head><body><div class="wrap">
<h1>metbit &middot; MTBLS1 Full-Pipeline Benchmark</h1>
<p class="sub">Raw Bruker FID &rarr; auto-phasing (ACME) &rarr; alignment &rarr; PQN normalisation &rarr; PCA &rarr; OPLS-DA &rarr; VIP &rarr; STOCSY.
 {perf['n_samples']} samples, {perf['n_features_normalised']:,} features. metbit v{env.get('metbit_version','?')}.</p>

<div class="kpis">
 <div class="kpi"><div class="v">{perf['total_wallclock_seconds']}s</div><div class="l">Total wall-clock</div></div>
 <div class="kpi"><div class="v">{opls['R2Y']:.3f}</div><div class="l">R2Y</div></div>
 <div class="kpi"><div class="v">{opls['Q2']:.3f}</div><div class="l">Q2</div></div>
 <div class="kpi"><div class="v">{outlier['n_after']}/{perf['n_samples']}</div><div class="l">Samples (after outlier drop)</div></div>
 <div class="kpi"><div class="v">{'yes' if repro['R2Y_max_abs_diff']==0 else 'no'}</div><div class="l">Deterministic (n={repro['n_repeats']})</div></div>
</div>

<h2>Spectra QC &mdash; auto-phasing</h2>
<figure><img src="data:image/png;base64,{spectra_png}">
<figcaption>All {perf['n_samples']} auto-phased spectra overlaid. Absorptive (upright) peaks on a flat baseline confirm correct phase; no dispersive lobes.</figcaption></figure>

<h2>Outlier exclusion</h2>
<table>
 <tr><td>Method</td><td>{outlier['method']}</td></tr>
 <tr><td>Samples in / out</td><td>{outlier['n_before']} &rarr; {outlier['n_after']} (excluded {outlier['n_excluded']})</td></tr>
 <tr><td>Hotelling T&sup2; limit</td><td>{outlier['T2_limit']}</td></tr>
 <tr><td>Excluded samples</td><td>{', '.join(outlier['excluded_samples']) or 'none'}</td></tr>
</table>

<h2>Multivariate results <span class="sub" style="font-weight:400">(after outlier exclusion)</span></h2>
<figure><img src="data:image/png;base64,{pca_png}"><figcaption>PCA scores, PC1 vs PC2, coloured by group (outliers already removed).</figcaption></figure>
<figure><img src="data:image/png;base64,{vip_png}"><figcaption>Top OPLS-DA VIP features. STOCSY anchor: {opls['anchor_ppm_top_vip']:.3f} ppm.</figcaption></figure>

<h2>Performance</h2>
<figure><img src="data:image/png;base64,{perf_png}"><figcaption>Per-stage wall-clock and peak memory (tracemalloc).</figcaption></figure>
<table><tr><th>Stage</th><th>Time (s)</th><th>Peak memory (MB)</th></tr>{perf_rows}
<tr><th>Total wall-clock</th><th>{perf['total_wallclock_seconds']}</th><th>&mdash;</th></tr></table>

<h2>Reproducibility <span class="sub" style="font-weight:400">(same-seed, n={repro['n_repeats']})</span></h2>
<table>
 <tr><th>Metric</th><th>Values across repeats</th><th>Max abs diff</th></tr>
 <tr><td>R2Y</td><td>{repro['R2Y_values']}</td><td>{repro['R2Y_max_abs_diff']}</td></tr>
 <tr><td>Q2</td><td>{repro['Q2_values']}</td><td>{repro['Q2_max_abs_diff']}</td></tr>
 <tr><td>Top-10 VIP ppm identical</td><td colspan="2">{repro['top10_VIP_ppm_identical_across_repeats']}</td></tr>
</table>

<h2>Usability</h2>
<table>
 <tr><td>Pipeline function length</td><td>{usab['pipeline_function_lines']} lines</td></tr>
 <tr><td>Distinct metbit API calls (raw FID &rarr; STOCSY)</td><td>{usab['distinct_metbit_api_calls_raw_fid_to_stocsy']}</td></tr>
 <tr><td>Manual intervention required</td><td>{usab['manual_intervention_required']}</td></tr>
</table>

<h2>Environment</h2>
<table>{env_rows}</table>

<p class="note">Disease status in MTBLS1 is confounded with acquisition batch and cross-validation is at the
 spectrum level, not participant level. R2Y/Q2/VIP are workflow-execution outputs, not evidence of disease discrimination.</p>
</div></body></html>"""

    out = RESULT / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
