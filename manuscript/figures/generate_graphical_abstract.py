"""
generate_graphical_abstract.py  v2
Publication-quality graphical abstract for the metbit manuscript.
Target: Bioinformatics Advances (OUP) – landscape TOC graphic
Output: graphical_abstract.png  (2400 x 900 px, 200 DPI)

Design principles
-----------------
- Two-accent palette (navy + teal) on white – no rainbow fills
- Realistic NMR / statistical schematics at each workflow stage
- Scientific notation only, zero code snippets
- Journal-standard typography (DejaVu Serif, 8-11 pt body)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Ellipse
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
WHITE   = "#FFFFFF"
BG      = "#FAFAFA"          # near-white canvas
PANEL   = "#FFFFFF"          # card background
NAVY    = "#1F4E79"          # primary – deep navy
STEEL   = "#2E75B6"          # secondary – steel blue
TEAL    = "#00847E"          # accent – teal
PURPLE  = "#5C2D91"          # modeling stages
AMBER   = "#B8860B"          # output stage
BORDER  = "#DEE2E6"          # card border
DARK    = "#2C3E50"          # body text
MID     = "#5D6D7E"          # sub-text
ARROW   = "#7F8C8D"          # connector arrows
RED     = "#C0392B"          # threshold / significance line
BLUE_PT = "#2E75B6"          # Group A points
ORAN_PT = "#E67E22"          # Group B points

STAGE_COLORS = [NAVY, STEEL, TEAL, STEEL, PURPLE, AMBER]
STAGE_LABELS = [
    "1  NMR FID Input",
    "2  Preprocessing",
    "3  Normalization\n& Alignment",
    "4  PCA",
    "5  Exploratory\nOPLS-DA",
    "6  VIP + STOCSY",
]
STAGE_SUBS = [
    "Raw FID (time domain)\nBruker acquisition",
    "FFT · Phase · Baseline\nCalibration",
    "PQN · MSC\nicoshift",
    "Unsupervised QC\nBatch / outlier detection",
    "Supervised exploration\nStudy-aware validation required",
    "Variable ranking\nSpectral connectivity",
]

# ── canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4.5), dpi=200)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis("off")
ax.set_facecolor(BG)


# ── utility helpers ───────────────────────────────────────────────────────────
def card(ax, x, y, w, h, accent, zorder=3):
    """White card with rounded corners and a thin accent top bar."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.18",
        linewidth=0.8, edgecolor=BORDER,
        facecolor=PANEL, zorder=zorder,
    )
    ax.add_patch(box)
    # coloured header strip
    hh = 0.52
    header = FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle="round,pad=0,rounding_size=0.18",
        linewidth=0, edgecolor=accent,
        facecolor=accent, zorder=zorder + 1, clip_on=False,
    )
    ax.add_patch(header)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.8, mutation_scale=14), zorder=6)


# ── NMR spectrum helper ───────────────────────────────────────────────────────
def lorentzian(x, x0, amp, width):
    return amp / (1 + ((x - x0) / width) ** 2)


def fid_signal(ax, cx, cy, w, h, color=NAVY):
    """Time-domain NMR Free Induction Decay (decaying oscillation)."""
    t  = np.linspace(0, 1, 700)
    xs = cx - w / 2 + t * w
    # multi-component FID: two dominant frequencies + decay envelope
    decay  = np.exp(-t * 3.8)
    signal = decay * (0.48 * np.sin(2 * np.pi * 9.0  * t) +
                      0.28 * np.sin(2 * np.pi * 14.5 * t) +
                      0.15 * np.sin(2 * np.pi * 22.0 * t))
    ys = cy + signal * h
    ax.fill_between(xs, cy, ys, alpha=0.20, color=color, zorder=7)
    ax.plot(xs, ys, color=color, lw=1.3, zorder=8)
    # time axis
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy],
            color=color, lw=0.7, alpha=0.40, zorder=7)
    # axis labels
    ax.text(cx + w / 2 + 0.03, cy, "t", ha="left", va="center",
            fontsize=6.0, color=color, style="italic", zorder=9)
    ax.text(cx - w / 2 - 0.01, cy + h * 0.55, "FID",
            ha="right", va="center",
            fontsize=5.8, color=color, zorder=9)


def nmr_spectrum(ax, cx, cy, w, h, color=NAVY, n_peaks=5, seed=0):
    """Realistic-looking 1H NMR spectrum (reversed ppm axis)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(cx - w / 2, cx + w / 2, 600)
    ys = np.zeros_like(xs)
    positions = rng.uniform(cx - w / 2 + 0.05, cx + w / 2 - 0.05, n_peaks)
    amps      = rng.uniform(0.15, 0.55, n_peaks)
    widths    = rng.uniform(0.012, 0.030, n_peaks)
    for p, a, ww in zip(positions, amps, widths):
        ys += lorentzian(xs, p, a, ww)
    ys = cy + ys * h
    baseline = cy + np.zeros_like(xs)
    ax.fill_between(xs, baseline, ys, alpha=0.18, color=color, zorder=7)
    ax.plot(xs, ys, color=color, lw=1.2, zorder=8)
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy],
            color=color, lw=0.7, alpha=0.4, zorder=7)


def fid_to_spectrum(ax, cx, cy, w, h, color=STEEL):
    """
    Preprocessing icon: FID (left) --FFT--> frequency-domain spectrum (right).
    Shows the core transformation clearly in a compact panel.
    """
    half  = w * 0.42      # half-width of each sub-panel
    gap   = w * 0.16      # gap between the two sub-panels (arrow area)
    cx_L  = cx - half / 2 - gap / 2   # center of FID sub-panel
    cx_R  = cx + half / 2 + gap / 2   # center of spectrum sub-panel
    sh    = h * 0.72                   # sub-panel height

    # ── left panel: decaying FID ──────────────────────────────────────────
    t  = np.linspace(0, 1, 400)
    xL = cx_L - half / 2 + t * half
    decay  = np.exp(-t * 4.5)
    signal = decay * (0.46 * np.sin(2 * np.pi * 8.0  * t) +
                      0.22 * np.sin(2 * np.pi * 14.0 * t))
    yL = cy + signal * sh
    ax.fill_between(xL, cy, yL, alpha=0.18, color=color, zorder=7)
    ax.plot(xL, yL, color=color, lw=1.1, zorder=8)
    ax.plot([xL[0], xL[-1]], [cy, cy], color=color, lw=0.6,
            alpha=0.35, zorder=7)
    ax.text(cx_L, cy - sh * 0.30, "FID", ha="center", va="top",
            fontsize=5.6, color=MID, zorder=9)

    # ── FFT arrow in the middle ───────────────────────────────────────────
    ax.annotate("", xy=(cx_R - half / 2 - 0.02, cy + sh * 0.12),
                xytext=(cx_L + half / 2 + 0.02, cy + sh * 0.12),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=12), zorder=9)
    ax.text(cx, cy + sh * 0.32, "FFT", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=color, zorder=10)

    # ── right panel: frequency-domain spectrum ────────────────────────────
    xR  = np.linspace(cx_R - half / 2, cx_R + half / 2, 400)
    yR  = np.zeros_like(xR)
    pks = [(cx_R - 0.24, 0.38, 0.016),
           (cx_R - 0.07, 0.56, 0.014),
           (cx_R + 0.10, 0.26, 0.015),
           (cx_R + 0.26, 0.42, 0.015)]
    for p, a, ww in pks:
        yR += lorentzian(xR, p, a, ww)
    yR_plot = cy + yR * sh
    ax.fill_between(xR, cy, yR_plot, alpha=0.18, color=color, zorder=7)
    ax.plot(xR, yR_plot, color=color, lw=1.1, zorder=8)
    ax.plot([xR[0], xR[-1]], [cy, cy], color=color, lw=0.6,
            alpha=0.35, zorder=7)
    ax.text(cx_R, cy - sh * 0.30, "Spectrum", ha="center", va="top",
            fontsize=5.6, color=MID, zorder=9)


def spectra_stack(ax, cx, cy, w, h, n=3):
    """Stacked aligned spectra for normalization stage."""
    colors = [TEAL, "#00A699", "#33C1BA"]
    for k in range(n):
        dy = k * (h * 0.38)
        xs = np.linspace(cx - w / 2, cx + w / 2, 500)
        ys = np.zeros_like(xs)
        peaks = [(cx - 0.28, 0.35, 0.022),
                 (cx + 0.05, 0.50, 0.018),
                 (cx + 0.30, 0.30, 0.020)]
        for p, a, ww in peaks:
            ys += lorentzian(xs, p, a, ww)
        ys = cy + dy + ys * h * 0.55
        ax.plot(xs, ys, color=colors[k], lw=1.0, alpha=0.85, zorder=7 + k)


def pca_scores(ax, cx, cy, r=0.40):
    """Two-cluster PCA scores schematic with confidence ellipses."""
    rng = np.random.default_rng(7)
    # Group A
    xa = rng.normal(cx - 0.18, 0.10, 22)
    ya = rng.normal(cy + 0.10, 0.09, 22)
    # Group B
    xb = rng.normal(cx + 0.18, 0.10, 22)
    yb = rng.normal(cy - 0.10, 0.09, 22)
    ax.scatter(xa, ya, color=BLUE_PT, s=8, zorder=8,
               edgecolors="white", linewidths=0.3, alpha=0.9)
    ax.scatter(xb, yb, color=ORAN_PT, s=8, zorder=8,
               edgecolors="white", linewidths=0.3, alpha=0.9)
    # ellipses
    for (mx, my, col) in [(np.mean(xa), np.mean(ya), BLUE_PT),
                          (np.mean(xb), np.mean(yb), ORAN_PT)]:
        ell = Ellipse((mx, my), width=0.34, height=0.28,
                      linewidth=0.8, edgecolor=col,
                      facecolor=col, alpha=0.12, zorder=7)
        ax.add_patch(ell)
    # axes lines
    aw = r * 1.05
    ax.annotate("", xy=(cx + aw, cy), xytext=(cx - aw, cy),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=0.6, mutation_scale=8), zorder=6)
    ax.annotate("", xy=(cx, cy + aw * 0.65), xytext=(cx, cy - aw * 0.65),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=0.6, mutation_scale=8), zorder=6)
    ax.text(cx + aw + 0.03, cy, "PC1\n52.3%", ha="left", va="center",
            fontsize=5.5, color=DARK, zorder=9)
    ax.text(cx, cy + aw * 0.65 + 0.03, "PC2\n18.7%", ha="center", va="bottom",
            fontsize=5.5, color=DARK, zorder=9)


def opls_scores(ax, cx, cy, r=0.38):
    """Exploratory OPLS-DA score plot."""
    rng = np.random.default_rng(11)
    xa = rng.normal(cx - 0.20, 0.08, 22)
    ya = rng.normal(cy,        0.11, 22)
    xb = rng.normal(cx + 0.20, 0.08, 22)
    yb = rng.normal(cy,        0.11, 22)
    ax.scatter(xa, ya, color=BLUE_PT, s=8, zorder=8,
               edgecolors="white", linewidths=0.3, alpha=0.9)
    ax.scatter(xb, yb, color=ORAN_PT, s=8, zorder=8,
               edgecolors="white", linewidths=0.3, alpha=0.9)
    for (mx, my, col) in [(np.mean(xa), cy, BLUE_PT),
                          (np.mean(xb), cy, ORAN_PT)]:
        ell = Ellipse((mx, my), width=0.30, height=0.32,
                      linewidth=0.8, edgecolor=col,
                      facecolor=col, alpha=0.12, zorder=7)
        ax.add_patch(ell)
    # axes
    ax.annotate("", xy=(cx + r + 0.02, cy), xytext=(cx - r - 0.02, cy),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=0.6, mutation_scale=8), zorder=6)
    ax.annotate("", xy=(cx, cy + r * 0.70), xytext=(cx, cy - r * 0.70),
                arrowprops=dict(arrowstyle="-|>", color=DARK,
                                lw=0.6, mutation_scale=8), zorder=6)
    ax.text(cx + r + 0.04, cy, "T_p", ha="left", va="center",
            fontsize=5.5, color=DARK, zorder=9, style="italic")
    ax.text(cx, cy + r * 0.70 + 0.03, "T_o", ha="center", va="bottom",
            fontsize=5.5, color=DARK, zorder=9, style="italic")
    # Interpretation annotation
    ax.text(cx, cy - r * 0.70 - 0.06,
            "Descriptive scores\nnot clinical validation", ha="center", va="top",
            fontsize=5.4, color=PURPLE, fontweight="bold", zorder=9)


def vip_stocsy(ax, cx, cy, w, h):
    """VIP bar chart above + STOCSY connectivity spectrum below."""
    # ── VIP bars (top half) ──
    vy = cy + h * 0.28
    n_bars = 8
    bw     = w / (n_bars * 1.55)
    vip    = np.array([2.45, 1.85, 0.72, 2.10, 0.55, 1.62, 0.90, 2.30])
    xs     = np.linspace(cx - w / 2 + bw * 0.5, cx + w / 2 - bw * 0.5, n_bars)
    bar_colors = [RED if v > 1.5 else MID for v in vip]
    for xi, vi, bc in zip(xs, vip, bar_colors):
        bar_h = vi / 3.0 * h * 0.38
        ax.bar(xi, bar_h, width=bw * 0.78, bottom=vy,
               color=bc, zorder=7, linewidth=0)
    # threshold line
    thr_h = 1.5 / 3.0 * h * 0.38
    ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04],
            [vy + thr_h, vy + thr_h],
            color=RED, lw=0.9, linestyle="--", zorder=8, alpha=0.8)
    ax.text(cx + w / 2 - 0.03, vy + thr_h + 0.01,
            "VIP>1.5", ha="right", va="bottom",
            fontsize=5.2, color=RED, zorder=9)

    # ── STOCSY spectrum (bottom half) ──
    sy   = cy + 0.04
    xs_s = np.linspace(cx - w / 2 + 0.04, cx + w / 2 - 0.04, 400)
    ys_s = np.zeros_like(xs_s)
    corr_peaks = [(cx - 0.20, 0.85, 0.025), (cx, 0.92, 0.022),
                  (cx + 0.22, 0.78, 0.024), (cx + 0.30, -0.62, 0.020)]
    for p, a, ww in corr_peaks:
        ys_s += lorentzian(xs_s, p, a, ww)
    ys_s_plot = sy + ys_s * h * 0.28
    # color by correlation sign
    pos = ys_s > 0
    ax.fill_between(xs_s, sy, ys_s_plot, where=pos,
                    color=TEAL, alpha=0.55, zorder=7)
    ax.fill_between(xs_s, sy, ys_s_plot, where=~pos,
                    color=RED, alpha=0.45, zorder=7)
    ax.plot(xs_s, ys_s_plot, color=DARK, lw=0.6, alpha=0.4, zorder=8)
    ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04], [sy, sy],
            color=DARK, lw=0.5, alpha=0.3, zorder=7)


# ── layout constants ──────────────────────────────────────────────────────────
N_STAGES  = 6
CARD_W    = 1.58
CARD_H    = 3.30
X_START   = 0.22
X_STEP    = (12 - X_START * 2 - CARD_W) / (N_STAGES - 1)
CARD_Y    = 0.65
HEADER_H  = 0.52
ICON_H    = 0.88    # height allocated to the schematic icon
ICON_CY   = CARD_Y + CARD_H * 0.40   # vertical center for icon

# ── title banner ──────────────────────────────────────────────────────────────
banner = FancyBboxPatch(
    (0.15, 4.05), 11.70, 0.35,
    boxstyle="round,pad=0,rounding_size=0.12",
    linewidth=0, facecolor=NAVY, zorder=3,
)
ax.add_patch(banner)
ax.text(6.0, 4.225,
        "metbit   ·   Scriptable ¹H NMR Metabolomics in Python",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color=WHITE, zorder=5,
        fontfamily="DejaVu Serif")

# ── stage cards and icons ─────────────────────────────────────────────────────
draw_fns = [
    lambda ax, cx, cy: fid_signal(ax, cx, cy, 1.22, ICON_H * 0.80,
                                  color=NAVY),
    lambda ax, cx, cy: fid_to_spectrum(ax, cx, cy, 1.28, ICON_H * 0.82,
                                       color=STEEL),
    lambda ax, cx, cy: spectra_stack(ax, cx, cy - 0.08, 1.20, ICON_H * 0.75),
    lambda ax, cx, cy: pca_scores(ax, cx, cy, r=0.40),
    lambda ax, cx, cy: opls_scores(ax, cx, cy, r=0.38),
    lambda ax, cx, cy: vip_stocsy(ax, cx, cy - 0.10, 1.28, ICON_H * 1.10),
]

for i in range(N_STAGES):
    cx = X_START + i * X_STEP + CARD_W / 2
    x0 = X_START + i * X_STEP

    # card
    card(ax, x0, CARD_Y, CARD_W, CARD_H, STAGE_COLORS[i])

    # header text
    ax.text(cx, CARD_Y + CARD_H - HEADER_H / 2,
            STAGE_LABELS[i],
            ha="center", va="center",
            fontsize=7.8, fontweight="bold", color=WHITE, zorder=6,
            multialignment="center", fontfamily="DejaVu Serif")

    # sub-label (method names)
    ax.text(cx, CARD_Y + 0.20,
            STAGE_SUBS[i],
            ha="center", va="center",
            fontsize=6.2, color=MID, zorder=6,
            multialignment="center", linespacing=1.4)

    # icon
    draw_fns[i](ax, cx, ICON_CY)

    # connecting arrow
    if i < N_STAGES - 1:
        ax_start = x0 + CARD_W + 0.04
        ax_end   = x0 + X_STEP - 0.04
        arrow(ax, ax_start, ax_end, CARD_Y + CARD_H / 2)

# ── footer bar ────────────────────────────────────────────────────────────────
footer = FancyBboxPatch(
    (0.15, 0.12), 11.70, 0.40,
    boxstyle="round,pad=0,rounding_size=0.10",
    linewidth=0, facecolor=BORDER, alpha=0.55, zorder=3,
)
ax.add_patch(footer)
ax.text(1.3, 0.32,
        "pip install metbit",
        ha="center", va="center",
        fontsize=7, color=NAVY, fontweight="bold",
        fontfamily="Courier New", zorder=5)
ax.text(6.0, 0.32,
        "github.com/aeiwz/metbit   ·   metbit-docs.vercel.app   ·   MIT License",
        ha="center", va="center",
        fontsize=7, color=DARK, zorder=5)
ax.text(11.2, 0.32,
        "v8.7.7",
        ha="center", va="center",
        fontsize=7, color=MID, zorder=5)

# ── Group A / B legend ────────────────────────────────────────────────────────
for col, label, xpos in [(BLUE_PT, "Group A", 9.28),
                         (ORAN_PT, "Group B", 9.92)]:
    ax.scatter([xpos], [0.32], color=col, s=18, zorder=6,
               edgecolors="white", linewidths=0.4)
    ax.text(xpos + 0.10, 0.32, label, va="center",
            fontsize=6.5, color=DARK, zorder=6)

# ── save ──────────────────────────────────────────────────────────────────────
import os
out = os.path.join(os.path.dirname(__file__), "graphical_abstract.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close()
