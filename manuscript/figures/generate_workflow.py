"""
generate_workflow.py  v2
Publication-quality analytical workflow figure for the metbit manuscript.
Output: workflow_diagram.png
Target: Bioinformatics Advances (OUP) – methods/overview figure

Design principles
-----------------
- Single neutral palette: navy accent, white cards, teal/purple/amber for stages
- No Python function calls – scientific method names only
- Each node shows data type in → data type out
- Proper fork/merge with explicit validation requirements
- Print-ready: 200 DPI, 7 × 14 inch canvas
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── palette ───────────────────────────────────────────────────────────────────
WHITE  = "#FFFFFF"
BG     = "#FAFAFA"
DARK   = "#2C3E50"
MID    = "#5D6D7E"
BORDER = "#DEE2E6"
LGRAY  = "#F2F3F4"

NAVY   = "#1F4E79"    # Data input
STEEL  = "#2E75B6"    # Preprocessing
TEAL   = "#00847E"    # Normalization
DTEAL  = "#006B65"    # Alignment
PURPLE = "#5C2D91"    # Modeling / validation
AMBER  = "#B8860B"    # Output / visualization
RED    = "#C0392B"    # significance / threshold
ARROW  = "#7F8C8D"

# Stage → accent-color mapping
C = {
    "input":   NAVY,
    "pre":     STEEL,
    "norm":    TEAL,
    "align":   DTEAL,
    "pca":     STEEL,
    "opls":    PURPLE,
    "valid":   PURPLE,
    "vip":     PURPLE,
    "stocsy":  TEAL,
    "output":  AMBER,
    "decision":RED,
}

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 7.2, 14.4
fig, ax = plt.subplots(figsize=(W, H), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ── layout constants ──────────────────────────────────────────────────────────
MX    = W / 2          # main column center x
MW    = 5.80           # main column width
MX0   = (W - MW) / 2  # main column left edge

FW    = (MW - 0.30) / 2    # fork column width
FX_L  = MX0                 # left fork left edge
FX_R  = MX0 + FW + 0.30    # right fork left edge
FCX_L = FX_L + FW / 2      # left fork center x
FCX_R = FX_R + FW / 2      # right fork center x

TITLE_FS  = 9.5
ITEM_FS   = 7.8
BADGE_FS  = 6.8
DT_FS     = 6.5
LINE_SP   = 1.45
PAD_V     = 0.22

# ── helpers ───────────────────────────────────────────────────────────────────
def node_height(items):
    return 0.52 + PAD_V + len(items) * 0.46 + PAD_V


def draw_node(x0, y_top, width, color, title, items,
              dtype_in=None, dtype_out=None, title_fs=TITLE_FS,
              item_fs=ITEM_FS, zorder=3):
    """
    Draw a workflow node.
    Returns the bottom y-coordinate.
    """
    h = node_height(items)
    y0 = y_top - h

    # shadow
    shd = FancyBboxPatch((x0 + 0.04, y0 - 0.04), width, h,
                         boxstyle="round,pad=0,rounding_size=0.14",
                         linewidth=0, facecolor="#CCCCCC",
                         alpha=0.30, zorder=zorder - 1)
    ax.add_patch(shd)

    # body
    body = FancyBboxPatch((x0, y0), width, h,
                          boxstyle="round,pad=0,rounding_size=0.14",
                          linewidth=0.7, edgecolor=BORDER,
                          facecolor=WHITE, zorder=zorder)
    ax.add_patch(body)

    # left accent bar
    bar = FancyBboxPatch((x0, y0), 0.22, h,
                         boxstyle="round,pad=0,rounding_size=0.10",
                         linewidth=0, facecolor=color,
                         zorder=zorder + 1)
    ax.add_patch(bar)

    # title
    ax.text(x0 + 0.32, y0 + h - 0.40, title,
            ha="left", va="center",
            fontsize=title_fs, fontweight="bold", color=DARK,
            zorder=zorder + 2)

    # items
    for k, it in enumerate(items):
        ty = y0 + h - 0.52 - PAD_V - (k + 0.5) * 0.46
        bullet_col = color if k == 0 else MID
        ax.text(x0 + 0.42, ty, "• " + it,
                ha="left", va="center",
                fontsize=item_fs, color=DARK,
                zorder=zorder + 2)

    # data-type annotations (right-aligned, small gray)
    if dtype_in:
        ax.text(x0 + width - 0.08, y0 + h - 0.22,
                "▲ " + dtype_in,
                ha="right", va="center",
                fontsize=DT_FS, color=MID, style="italic",
                zorder=zorder + 2)
    if dtype_out:
        ax.text(x0 + width - 0.08, y0 + 0.14,
                "▼ " + dtype_out,
                ha="right", va="center",
                fontsize=DT_FS, color=color, style="italic",
                zorder=zorder + 2)

    return y0


def draw_arrow(x, y_top, y_bot, lw=1.6):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=lw, mutation_scale=13), zorder=5)


def draw_diamond(x0, y_top, width, height, color, label, sublabel=None):
    """Decision diamond for validation step."""
    cx = x0 + width / 2
    cy = y_top - height / 2
    dx, dy = width / 2, height / 2
    xs = [cx,      cx + dx, cx,      cx - dx, cx]
    ys = [cy + dy, cy,      cy - dy, cy,      cy + dy]
    ax.fill(xs, ys, color=color, alpha=0.12, zorder=3)
    ax.plot(xs, ys, color=color, lw=1.2, zorder=4)
    ax.text(cx, cy, label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=color, zorder=5)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel,
                ha="center", va="center",
                fontsize=7.0, color=MID, zorder=5)
    return cy - dy   # return bottom y


def h_line(x0, x1, y, col=ARROW, lw=1.6):
    ax.plot([x0, x1], [y, y], color=col, lw=lw, zorder=5)


def v_line(x, y0, y1, col=ARROW, lw=1.6):
    ax.plot([x, x], [y0, y1], color=col, lw=lw, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# TITLE BANNER
# ─────────────────────────────────────────────────────────────────────────────
tb = FancyBboxPatch((MX0, H - 1.10), MW, 0.80,
                    boxstyle="round,pad=0,rounding_size=0.18",
                    linewidth=0, facecolor=NAVY, zorder=3)
ax.add_patch(tb)
ax.text(MX, H - 0.70, "metbit  ·  Analytical Workflow",
        ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=WHITE, zorder=5,
        fontfamily="DejaVu Serif")
ax.text(MX, H - 0.98, "¹H NMR Metabolomics  ·  Local Scriptable Workflow",
        ha="center", va="center",
        fontsize=8.0, color="#A9CCE3", zorder=5)

y = H - 1.12   # running y cursor (top of current box)
GAP = 0.28     # gap between boxes

# ─────────────────────────────────────────────────────────────────────────────
# 1  DATA INPUT
# ─────────────────────────────────────────────────────────────────────────────
y -= GAP
y = draw_node(
    MX0, y, MW, C["input"],
    "1  NMR FID Input",
    ["Raw Free Induction Decay (FID) files from Bruker acquisition",
     "Sample metadata table (group labels, covariates)",
     "Vendor-neutral: pre-processed sample × variable matrix also accepted"],
    dtype_in="Raw NMR FID (time domain)",
    dtype_out="pandas DataFrame (samples × ppm)",
)
draw_arrow(MX, y, y - GAP)
y -= GAP

# ─────────────────────────────────────────────────────────────────────────────
# 2  NMR PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
y = draw_node(
    MX0, y, MW, C["pre"],
    "2  NMR Preprocessing",
    ["Digital filter removal + zero-filling",
     "Fast Fourier Transform (FFT): time domain → frequency domain",
     "Automated phase correction (entropy minimization)",
     "Baseline correction (arPLS / AirPLS)",
     "Chemical-shift calibration to internal reference (TSP/DSS = 0.00 ppm)"],
    dtype_in="Complex FID array",
    dtype_out="Real-valued spectral matrix",
)
draw_arrow(MX, y, y - GAP)
y -= GAP

# ─────────────────────────────────────────────────────────────────────────────
# 3  NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
y = draw_node(
    MX0, y, MW, C["norm"],
    "3  Spectral Normalization",
    ["Probabilistic Quotient Normalization (PQN) — dilution correction",
     "Multiplicative Scatter Correction (MSC) — scatter suppression",
     "Total Spectral Area (TSA) — integral-based scaling"],
    dtype_in="Raw spectral matrix",
    dtype_out="Normalized spectral matrix",
)
draw_arrow(MX, y, y - GAP)
y -= GAP

# ─────────────────────────────────────────────────────────────────────────────
# 4  PEAK ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
y = draw_node(
    MX0, y, MW, C["align"],
    "4  Chemical-Shift Alignment",
    ["Interval correlation-optimized shifting (icoshift)",
     "Maximizes inter-sample spectral correlations within defined intervals",
     "scikit-learn–compatible transformer interface"],
    dtype_in="Normalized spectral matrix",
    dtype_out="Aligned spectral matrix",
)

# ─────────────────────────────────────────────────────────────────────────────
# FORK  ──  PCA (left)  and  OPLS-DA (right)
# ─────────────────────────────────────────────────────────────────────────────
FORK_Y = y - GAP * 0.6
v_line(MX, y, FORK_Y)
h_line(FCX_L, FCX_R, FORK_Y)
draw_arrow(FCX_L, FORK_Y, FORK_Y - GAP * 0.4, lw=1.4)
draw_arrow(FCX_R, FORK_Y, FORK_Y - GAP * 0.4, lw=1.4)

y_fork = FORK_Y - GAP * 0.4

# ── left branch: PCA ─────────────────────────────────────────────────────────
yL = draw_node(
    FX_L, y_fork, FW, C["pca"],
    "5a  PCA",
    ["Unsupervised variance decomposition",
     "Scores plot (PC1 vs PC2)",
     "Outlier detection & batch inspection",
     "Cumulative explained variance (scree)"],
    dtype_in="Aligned matrix",
    dtype_out="Scores, loadings",
    title_fs=8.8, item_fs=7.2,
)

# ── right branch: OPLS-DA ─────────────────────────────────────────────────────
yR = draw_node(
    FX_R, y_fork, FW, C["opls"],
    "5b  Exploratory OPLS-DA",
    ["Binary group modeling",
     "Predictive vs orthogonal separation",
     "Sample-level k-fold diagnostics (R²X, R²Y, Q²)",
     "Study-aware validation required for inference"],
    dtype_in="Aligned matrix + labels",
    dtype_out="Model object (T_p, T_o)",
    title_fs=8.8, item_fs=7.2,
)

# ── validation requirements (right branch continues) ────────────────────────
GAP_F = 0.22
draw_arrow(FCX_R, yR, yR - GAP_F, lw=1.4)
yR = draw_node(
    FX_R, yR - GAP_F, FW, C["valid"],
    "Validation Requirements",
    ["Participant-grouped data splitting",
     "Preprocessing fitted within training folds",
     "Permutation of the complete pipeline and score"],
    dtype_in="Study design + model",
    dtype_out="Defensible validation plan",
    title_fs=8.8, item_fs=7.2,
)

# ── VIP scoring (right branch continues) ─────────────────────────────────────
draw_arrow(FCX_R, yR, yR - GAP_F, lw=1.4)
yR = draw_node(
    FX_R, yR - GAP_F, FW, C["vip"],
    "Variable Importance",
    ["VIP scores for variable ranking",
     "S-plot (covariance vs correlation)",
     "Independent structural confirmation required"],
    dtype_in="OPLS-DA loadings",
    dtype_out="Ranked variable list",
    title_fs=8.8, item_fs=7.2,
)

# ── STOCSY (left branch picks up VIP output) ─────────────────────────────────
draw_arrow(FCX_L, yL, yL - GAP_F, lw=1.4)
yL2 = draw_node(
    FX_L, yL - GAP_F, FW, C["stocsy"],
    "STOCSY",
    ["Pearson correlation vs anchor resonance",
     "Connectivity spectrum (r color-coded)",
     "Metabolite spin-system deconvolution",
     "Interactive Dash explorer"],
    dtype_in="VIP anchor ppm",
    dtype_out="Connectivity spectrum",
    title_fs=8.8, item_fs=7.2,
)

# ─────────────────────────────────────────────────────────────────────────────
# MERGE  ──  both branches converge to output
# ─────────────────────────────────────────────────────────────────────────────
MERGE_Y = min(yL2, yR) - GAP * 0.5
v_line(FCX_L, yL2, MERGE_Y)
v_line(FCX_R, yR, MERGE_Y)
h_line(FCX_L, FCX_R, MERGE_Y)
draw_arrow(MX, MERGE_Y, MERGE_Y - GAP * 0.4, lw=1.8)

y = MERGE_Y - GAP * 0.4

# ─────────────────────────────────────────────────────────────────────────────
# 6  INTERACTIVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
y = draw_node(
    MX0, y, MW, C["output"],
    "6  Interactive Output & Export",
    ["Plotly HTML figures — scores, loadings, VIP, S-plot, trajectory",
     "Local Dash applications — STOCSY explorer, peak annotation",
     "pandas DataFrame export (VIP scores, STOCSY correlations)",
     "PNG / SVG / HTML — publication-ready figures"],
    dtype_in="All analysis objects",
    dtype_out="Figures · Tables · Reports",
)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
LEG_ITEMS = [
    (C["input"],  "Spectral Input"),
    (C["pre"],    "Preprocessing"),
    (C["norm"],   "Normalization"),
    (C["align"],  "Alignment"),
    (C["opls"],   "Statistical Modeling"),
    (C["output"], "Output"),
]
LEG_Y  = y - 0.55
LEG_SW = 0.20
LEG_SH = 0.18
item_w = MW / len(LEG_ITEMS)
for k, (col, lbl) in enumerate(LEG_ITEMS):
    lx = MX0 + k * item_w + (item_w - LEG_SW) / 2 - 0.15
    leg_box = FancyBboxPatch((lx, LEG_Y), LEG_SW, LEG_SH,
                              boxstyle="round,pad=0,rounding_size=0.05",
                              linewidth=0, facecolor=col, zorder=4)
    ax.add_patch(leg_box)
    ax.text(lx + LEG_SW + 0.07, LEG_Y + LEG_SH / 2, lbl,
            ha="left", va="center",
            fontsize=6.5, color=DARK, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER  ─  DOI / URL badge
# ─────────────────────────────────────────────────────────────────────────────
FOOT_Y = LEG_Y - 0.45
ax.text(MX, FOOT_Y,
        "github.com/aeiwz/metbit   ·   metbit-docs.vercel.app   ·   MIT License",
        ha="center", va="center",
        fontsize=7.5, color=MID, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "workflow_diagram.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close()
