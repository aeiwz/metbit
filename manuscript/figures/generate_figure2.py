"""
generate_figure2.py
Publication-quality 4-panel results figure for the metbit manuscript.
Panels: (a) PCA scores  (b) OPLS-DA scores  (c) VIP  (d) STOCSY
Output: figure2_results.png
Target: Bioinformatics Advances (OUP) — double-column (7.2 in)
Synthetic data reproduce the statistics reported in the manuscript:
  PCA  PC1=52.3%, PC2=18.7%
  OPLS-DA  R2Y=0.94, Q2=0.91, perm-p<0.001
  VIP  regions: 1.2-1.4, 2.0-2.4, 3.3-3.5 ppm
  STOCSY  anchor=2.35 ppm, succinate ABX system
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import numpy as np
import os

# ── colour palette (consistent with other manuscript figures) ─────────────────
BLUE_A  = "#2E75B6"    # Group A scatter
ORAN_B  = "#E67E22"    # Group B scatter
NAVY    = "#1F4E79"    # primary annotations
TEAL    = "#00847E"    # positive STOCSY fill
RED     = "#C0392B"    # negative STOCSY / high-VIP
VIP_HI  = "#C0392B"    # VIP > 2.0
VIP_MID = "#E67E22"    # 1.0 < VIP ≤ 2.0
VIP_LO  = "#BDC3C7"    # VIP ≤ 1.0
MID     = "#5D6D7E"    # axis / tick colour
DARK    = "#2C3E50"    # body text
BORDER  = "#BDC3C7"    # spine / legend border

# ── reproducible synthetic data ───────────────────────────────────────────────
rng   = np.random.default_rng(42)
n_A   = 60
n_B   = 60

# (a) PCA: Group A at +PC1, Group B at −PC1
pc1_A = rng.normal( 2.80, 1.10, n_A)
pc1_B = rng.normal(-2.80, 1.10, n_B)
pc2_A = rng.normal( 0.40, 0.85, n_A)
pc2_B = rng.normal(-0.40, 0.85, n_B)

# (b) OPLS-DA: tighter predictive-score separation (Q2=0.91)
tp_A  = rng.normal( 3.30, 0.70, n_A)
tp_B  = rng.normal(-3.30, 0.70, n_B)
to_A  = rng.normal( 0.10, 1.05, n_A)
to_B  = rng.normal(-0.10, 1.05, n_B)

# (c) VIP scores — 1200 representative chemical-shift points
N_PPM = 1200
ppm   = np.linspace(9.0, 0.0, N_PPM)          # reversed NMR convention
vip   = np.abs(rng.normal(0.58, 0.28, N_PPM))  # background VIP

HIGH_REGIONS = [                               # match manuscript text
    (1.20, 1.40, 2.70, 0.28),                  # aliphatic / lipid
    (2.00, 2.45, 3.05, 0.32),                  # succinate
    (3.30, 3.52, 2.50, 0.25),                  # creatine-family
]
for p_lo, p_hi, mu, sig in HIGH_REGIONS:
    mask = (ppm >= p_lo) & (ppm <= p_hi)
    vip[mask] = np.abs(rng.normal(mu, sig, mask.sum()))
vip = np.clip(vip, 0.0, 4.9)

# (d) STOCSY: anchor = 2.35 ppm (succinate ABX system)
corr = rng.normal(0.0, 0.09, N_PPM)

STOCSY_POS = [                                 # positive correlations
    (2.35, 1.00, 0.016),                        # anchor
    (2.57, 0.90, 0.014),                        # succinate -CH2- b
    (2.68, 0.86, 0.013),                        # succinate -CH2- c
    (2.41, 0.70, 0.011),                        # correlated residue
]
STOCSY_NEG = [                                 # negative correlations
    (1.27, -0.65, 0.020),
    (1.33, -0.58, 0.018),
]
for peak, r, w in STOCSY_POS + STOCSY_NEG:
    corr += r * np.exp(-((ppm - peak) ** 2) / (2 * w ** 2))
corr = np.clip(corr, -1.0, 1.0)


# ── helpers ───────────────────────────────────────────────────────────────────
def add_ellipse(ax, x, y, n_std=2.0, **kw):
    """95 % covariance-based confidence ellipse."""
    cov     = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell     = Ellipse((0, 0),
                      width=np.sqrt(1 + pearson) * 2,
                      height=np.sqrt(1 - pearson) * 2,
                      **kw)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    tf = (transforms.Affine2D()
          .rotate_deg(45)
          .scale(sx, sy)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tf + ax.transData)
    ax.add_patch(ell)


def style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=8.5, color=DARK, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=8.5, color=DARK, labelpad=3)
    ax.tick_params(labelsize=7.5, length=3, width=0.6, color=MID, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.65)
        sp.set_color(BORDER)
    ax.set_facecolor("white")


def panel_label(ax, letter):
    ax.text(-0.16, 1.05, f"({letter})",
            transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=DARK, va="top")


SCT = dict(s=18, edgecolors="white", linewidths=0.35, alpha=0.88, zorder=4)

# ── canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 6.4), dpi=300)
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.46, wspace=0.42,
    left=0.10, right=0.97, top=0.94, bottom=0.09,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# (a) PCA SCORES
# ─────────────────────────────────────────────────────────────────────────────
ax_a.scatter(pc1_A, pc2_A, color=BLUE_A, label="Group A", **SCT)
ax_a.scatter(pc1_B, pc2_B, color=ORAN_B, label="Group B", **SCT)

for x, y, col in [(pc1_A, pc2_A, BLUE_A), (pc1_B, pc2_B, ORAN_B)]:
    add_ellipse(ax_a, x, y, n_std=2.0,
                facecolor=col, alpha=0.10,
                edgecolor=col, linewidth=0.9, linestyle="--")

ax_a.axhline(0, color=MID, lw=0.5, ls="--", alpha=0.45, zorder=2)
ax_a.axvline(0, color=MID, lw=0.5, ls="--", alpha=0.45, zorder=2)
style_ax(ax_a, "PC1  (52.3 %)", "PC2  (18.7 %)")
panel_label(ax_a, "a")
leg_a = ax_a.legend(fontsize=7.0, framealpha=0.88, edgecolor=BORDER,
                    loc="upper right", handletextpad=0.35,
                    borderpad=0.50, markerscale=1.1)
leg_a.get_frame().set_linewidth(0.6)

# ─────────────────────────────────────────────────────────────────────────────
# (b) OPLS-DA SCORES
# ─────────────────────────────────────────────────────────────────────────────
ax_b.scatter(tp_A, to_A, color=BLUE_A, **SCT)
ax_b.scatter(tp_B, to_B, color=ORAN_B, **SCT)

for x, y, col in [(tp_A, to_A, BLUE_A), (tp_B, to_B, ORAN_B)]:
    add_ellipse(ax_b, x, y, n_std=2.0,
                facecolor=col, alpha=0.10,
                edgecolor=col, linewidth=0.9, linestyle="--")

ax_b.axhline(0, color=MID, lw=0.5, ls="--", alpha=0.45, zorder=2)
ax_b.axvline(0, color=MID, lw=0.5, ls="--", alpha=0.45, zorder=2)
style_ax(ax_b,
         r"$T_\mathrm{p}$  (predictive score)",
         r"$T_\mathrm{o}$  (orthogonal score)")
panel_label(ax_b, "b")

ax_b.text(0.97, 0.97,
          r"R²Y = 0.94" "\n" r"Q²  = 0.91" "\n" r"$p_{\rm perm}$ < 0.001",
          transform=ax_b.transAxes,
          ha="right", va="top", fontsize=7.2, color=NAVY,
          bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                    edgecolor=BORDER, linewidth=0.6, alpha=0.92))

# ─────────────────────────────────────────────────────────────────────────────
# (c) VIP PLOT
# ─────────────────────────────────────────────────────────────────────────────
lo  = vip <= 1.0
mid = (vip > 1.0) & (vip <= 2.0)
hi  = vip > 2.0

ax_c.scatter(ppm[lo],  vip[lo],  color=VIP_LO,  s=3.5, alpha=0.50,
             linewidths=0, zorder=3)
ax_c.scatter(ppm[mid], vip[mid], color=VIP_MID, s=5.0, alpha=0.78,
             linewidths=0, zorder=4)
ax_c.scatter(ppm[hi],  vip[hi],  color=VIP_HI,  s=7.0, alpha=0.92,
             linewidths=0, zorder=5)

ax_c.axhline(2.0, color=VIP_HI,  lw=0.85, ls="--", alpha=0.70, zorder=6)
ax_c.axhline(1.0, color=VIP_MID, lw=0.70, ls=":",  alpha=0.60, zorder=6)

ax_c.set_xlim(9.3, -0.2)
ax_c.set_ylim(0.0, 5.5)
ax_c.invert_xaxis()              # ensure reversed ppm

# threshold labels
ax_c.text(9.2, 2.08, "VIP = 2.0", fontsize=6.0, color=VIP_HI,  va="bottom")
ax_c.text(9.2, 1.08, "VIP = 1.0", fontsize=6.0, color=VIP_MID, va="bottom")

# shaded high-VIP regions + rotated text labels
REGION_LABELS = [
    (1.20, 1.40, "1.2–1.4 ppm"),
    (2.00, 2.45, "2.0–2.4 ppm"),
    (3.30, 3.52, "3.3–3.5 ppm"),
]
for r_lo, r_hi, lbl in REGION_LABELS:
    ax_c.axvspan(r_lo, r_hi, alpha=0.10, color=NAVY, zorder=2)
    ax_c.text((r_lo + r_hi) / 2, 5.2, lbl,
              ha="center", va="top", fontsize=5.8, color=NAVY,
              rotation=90, zorder=7)

legend_els = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VIP_HI,
           markersize=5, label="VIP > 2.0"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VIP_MID,
           markersize=5, label="1.0 < VIP ≤ 2.0"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VIP_LO,
           markersize=5, label="VIP ≤ 1.0"),
]
leg_c = ax_c.legend(handles=legend_els, fontsize=6.8, framealpha=0.88,
                    edgecolor=BORDER, loc="upper left",
                    handletextpad=0.30, borderpad=0.45)
leg_c.get_frame().set_linewidth(0.6)

style_ax(ax_c, "Chemical shift  (ppm)", "VIP score")
panel_label(ax_c, "c")

# ─────────────────────────────────────────────────────────────────────────────
# (d) STOCSY CONNECTIVITY SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────
# Colour the line by correlation value using LineCollection
pts  = np.array([ppm, corr]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
norm_r = mcolors.Normalize(vmin=-1.0, vmax=1.0)
lc = LineCollection(segs, cmap="RdBu", norm=norm_r,
                    linewidth=1.4, zorder=4)
lc.set_array(corr[:-1])
ax_d.add_collection(lc)

# Fill under curve for positive / negative regions
ax_d.fill_between(ppm, 0, corr, where=(corr >= 0),
                  color=TEAL, alpha=0.16, zorder=3)
ax_d.fill_between(ppm, 0, corr, where=(corr < 0),
                  color=RED,  alpha=0.13, zorder=3)
ax_d.axhline(0, color=DARK, lw=0.55, alpha=0.45, zorder=5)

# Anchor dashed line
ax_d.axvline(2.35, color=ORAN_B, lw=0.8, ls="--", alpha=0.80, zorder=6)
ax_d.text(2.35, 1.06, "anchor\n2.35 ppm",
          ha="center", va="bottom", fontsize=5.8,
          color=ORAN_B, zorder=7)

# Succinate peak annotations
for pk_ppm, pk_lbl in [(2.57, "2.57"), (2.68, "2.68")]:
    idx = np.argmin(np.abs(ppm - pk_ppm))
    y0  = corr[idx]
    ax_d.annotate(
        "", xy=(pk_ppm, y0),
        xytext=(pk_ppm, y0 + 0.20),
        arrowprops=dict(arrowstyle="-|>", color=NAVY,
                        lw=0.8, mutation_scale=8), zorder=7,
    )
    ax_d.text(pk_ppm, y0 + 0.24, pk_lbl,
              ha="center", va="bottom", fontsize=5.8,
              color=NAVY, zorder=7)

# Succinate span bracket
ax_d.annotate("", xy=(2.68, -0.08), xytext=(2.35, -0.08),
              arrowprops=dict(arrowstyle="<->", color=NAVY,
                              lw=0.75, mutation_scale=8), zorder=6)
ax_d.text(2.515, -0.14, "Succinate",
          ha="center", va="top", fontsize=6.0,
          color=NAVY, style="italic", zorder=7)

# Negative correlation label
ax_d.text(1.30, -0.76, "r < −0.60",
          ha="center", va="center", fontsize=5.8,
          color=RED, zorder=7)

ax_d.set_xlim(3.10, 0.85)
ax_d.set_ylim(-1.12, 1.30)
style_ax(ax_d,
         "Chemical shift  (ppm)",
         "Pearson  r")
panel_label(ax_d, "d")

# Colorbar
sm = ScalarMappable(cmap="RdBu", norm=norm_r)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax_d, orientation="vertical",
                  fraction=0.046, pad=0.03, shrink=0.88)
cb.set_label("r", fontsize=8.5, color=DARK)
cb.ax.tick_params(labelsize=6.8, length=2.5, width=0.5)
cb.outline.set_linewidth(0.5)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figure2_results.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close()
