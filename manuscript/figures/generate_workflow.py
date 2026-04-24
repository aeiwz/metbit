"""
Detailed data-analysis workflow diagram for the metbit manuscript.
Output: workflow_diagram.png
Fixes: no overlapping side-labels, all text fits within boxes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#F7F9FC"
DARK    = "#1A1A2E"
WHITE   = "#FFFFFF"
C_INPUT = "#154360"
C_PRE   = "#1A5276"
C_NORM  = "#0E6655"
C_ALIGN = "#117A65"
C_STAT  = "#6C3483"
C_VIZ   = "#935116"
C_ARROW = "#626567"
C_SHADE = "#D5D8DC"

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 12, 30          # Increased height to 30
fig, ax = plt.subplots(figsize=(W * 0.72, H * 0.72), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ── layout constants ──────────────────────────────────────────────────────────
X0   = 0.55            # left edge of main boxes
WW   = W - 1.1         # width  of main boxes  (10.9)
CX   = W / 2           # horizontal centre

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec=WHITE, lw=1.0, r=0.28, alpha=1.0, zorder=3):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)


def arrow_down(x, y_top, y_bot, lw=2.2):
    ax.annotate(
        "", xy=(x, y_bot), xytext=(x, y_top),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW, lw=lw, mutation_scale=14,
        ), zorder=5,
    )


def arrow_h(x0, x1, y, lw=2.0):
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW, lw=lw, mutation_scale=12,
        ), zorder=5,
    )


TITLE_H   = 0.90   # height of the coloured title strip (increased for centered badge)
ITEM_STEP = 0.52   # vertical spacing between bullet lines
PAD_TOP   = 0.25   # gap between title strip and first bullet
PAD_BOT   = 0.25   # extra space below last bullet


def node(x, y, w, color, title, items,
         badge=None,
         title_fs=10.5, item_fs=8.5):
    """
    Draw a rounded box whose height is computed automatically.
    Returns the bottom y-coordinate of the box.
    """
    n      = len(items)
    box_h  = TITLE_H + PAD_TOP + n * ITEM_STEP + PAD_BOT
    y_bot  = y - box_h          # boxes are placed top-down

    # shadow
    rbox(x + 0.07, y_bot - 0.07, w, box_h, C_SHADE, ec=C_SHADE, lw=0,
         alpha=0.35, zorder=2)
    # body
    rbox(x, y_bot, w, box_h, color, r=0.26, zorder=3)
    # title strip
    rbox(x, y_bot + box_h - TITLE_H, w, TITLE_H, color, ec=WHITE,
         lw=0, r=0.26, zorder=4)

    # title text - positioned at the top of the header area
    ax.text(x + w / 2, y_bot + box_h - 0.32,
            title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=WHITE, zorder=6)

    # badge (dependency tag) - centered horizontally and placed below title
    if badge:
        bw, bh = 2.00, 0.32
        bx = x + (w - bw) / 2
        by = y_bot + box_h - 0.75
        rbox(bx, by, bw, bh, WHITE, ec=color, lw=1.0, r=0.12, zorder=7)
        ax.text(bx + bw / 2, by + bh / 2, badge,
                ha="center", va="center",
                fontsize=7.2, color=color, fontweight="bold", zorder=8)

    # bullet items
    for i, it in enumerate(items):
        ty = y_bot + box_h - TITLE_H - PAD_TOP - (i + 0.5) * ITEM_STEP
        ax.text(x + w / 2, ty, it,
                ha="center", va="center",
                fontsize=item_fs, color=WHITE, zorder=6)

    return y_bot          # return bottom for arrow placement


# ─────────────────────────────────────────────────────────────────────────────
# TITLE BANNER
# ─────────────────────────────────────────────────────────────────────────────
TITLE_BOX_TOP = H - 0.35
rbox(X0, TITLE_BOX_TOP - 0.85, WW, 0.85, C_INPUT, r=0.30, zorder=3)
ax.text(CX, TITLE_BOX_TOP - 0.85 / 2,
        "metbit  ─  Data Analysis Workflow",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=WHITE, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 1 DATA INPUT
# ─────────────────────────────────────────────────────────────────────────────
y_cursor = TITLE_BOX_TOP - 0.85 - 0.30   # top of next box

y_bot1 = node(
    X0, y_cursor, WW, C_INPUT,
    "1 Data Input",
    [
        "Bruker FID directory tree   ·   nmr_preprocessing class",
        "ng.bruker.read()  →  dic (acquisition params)  +  raw FID array",
    ],
    badge="nmrglue",
)

arrow_down(CX, y_bot1, y_bot1 - 0.30)
y_cursor = y_bot1 - 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 2 SIGNAL PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
y_bot2 = node(
    X0, y_cursor, WW, C_PRE,
    "2 Signal Preprocessing",
    [
        "Digital filter removal  ·  ng.bruker.remove_digital_filter()",
        "Zero-filling  →  Fourier Transform  (FFT)",
        "Auto phase correction  ·  peak_minima algorithm  (p0, p1)",
        "Baseline correction  ·  AsLS / AirPLS / ArPLS / ModPoly / rubberband",
        "Chemical-shift calibration  ·  TSP / DSS reference  (δ = 0.00 ppm)",
    ],
    badge="baseline.py",
)

arrow_down(CX, y_bot2, y_bot2 - 0.30)
y_cursor = y_bot2 - 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 3 SPECTRAL NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
y_bot3 = node(
    X0, y_cursor, WW, C_NORM,
    "3 Spectral Normalization",
    [
        "Probabilistic Quotient Normalization (PQN)  ·  Normalization.pqn_normalization()",
        "Standard Normal Variate (SNV)  ·  Normalization.snv_normalization()",
        "Multiplicative Scatter Correction (MSC)  ·  Normalization.msc_normalization()",
    ],
    badge="spec_norm.py",
)

arrow_down(CX, y_bot3, y_bot3 - 0.30)
y_cursor = y_bot3 - 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 4 PEAK ALIGNMENT & DETECTION
# ─────────────────────────────────────────────────────────────────────────────
y_bot4 = node(
    X0, y_cursor, WW, C_ALIGN,
    "4 Peak Alignment & Detection",
    [
        "icoshift_align()  ·  interval-correlation-optimized shifting (icoshift)",
        "detect_multiplets()  ·  singlet / doublet / triplet / quartet / multiplet",
        "PeakAligner  ·  scikit-learn-compatible  fit_transform() interface",
    ],
    badge="alignment.py",
)

# ─────────────────────────────────────────────────────────────────────────────
# FORK  ── STOCSY (left)  and  MULTIVARIATE (right)
# ─────────────────────────────────────────────────────────────────────────────
FORK_GAP  = 0.40    # gap between the two fork boxes
FW        = (WW - FORK_GAP) / 2    # each fork box width  (~5.25)
FX_L      = X0                      # left fork x
FX_R      = X0 + FW + FORK_GAP     # right fork x
FCX_L     = FX_L + FW / 2          # left centre x
FCX_R     = FX_R + FW / 2          # right centre x

# connector from ④ → fork
FORK_TOP = y_bot4 - 0.30
ax.plot([CX, CX], [y_bot4, FORK_TOP], color=C_ARROW, lw=2.2, zorder=5)
ax.plot([FCX_L, FCX_R], [FORK_TOP, FORK_TOP], color=C_ARROW, lw=2.2, zorder=5)
# drop-arrow to left box
ax.annotate("", xy=(FCX_L, FORK_TOP - 0.28), xytext=(FCX_L, FORK_TOP),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                            lw=2.0, mutation_scale=12), zorder=5)
# drop-arrow to right box
ax.annotate("", xy=(FCX_R, FORK_TOP - 0.28), xytext=(FCX_R, FORK_TOP),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                            lw=2.0, mutation_scale=12), zorder=5)

y_fork_cursor = FORK_TOP - 0.28

# ── 5a STOCSY ────────────────────────────────────────────────────────────────
y_botL1 = node(
    FX_L, y_fork_cursor, FW, C_STAT,
    "5a  STOCSY",
    [
        "STOCSY()  ·  Pearson corr. vs anchor ppm",
        "p-value threshold  (default p < 0.0001)",
        "STOCSY_app  ·  interactive Dash application",
    ],
    badge="STOCSY.py",
    title_fs=9.5, item_fs=8.0,
)

# ── 5b MULTIVARIATE MODELING ─────────────────────────────────────────────────
y_botR1 = node(
    FX_R, y_fork_cursor, FW, C_STAT,
    "5b  Multivariate Modeling",
    [
        "pca  ·  pareto / UV / mean-center / min-max",
        "opls_da  ·  OPLS-DA  +  permutation test",
        "lazy_opls_da  ·  automated pairwise OPLS-DA",
    ],
    badge="metbit.py",
    title_fs=9.5, item_fs=8.0,
)

# arrows within each branch
y_between = min(y_botL1, y_botR1) - 0.30

# ── STOCSY output ─────────────────────────────────────────────────────────────
arrow_down(FCX_L, y_botL1, y_botL1 - 0.30)
y_botL2 = node(
    FX_L, y_botL1 - 0.30, FW, C_VIZ,
    "STOCSY Output",
    [
        "Correlation spectrum  (colour-coded by r)",
        "Significant co-varying resonances",
        "pickie_peak  ·  interactive peak picker",
    ],
    title_fs=9.5, item_fs=8.0,
)

# ── MODEL DIAGNOSTICS ─────────────────────────────────────────────────────────
arrow_down(FCX_R, y_botR1, y_botR1 - 0.30)
y_botR2 = node(
    FX_R, y_botR1 - 0.30, FW, C_STAT,
    "Model Diagnostics",
    [
        "Cross-validation  (k-fold Q²)",
        "Permutation test  ·  n_permutations = 500",
        "R²X / R²Y / Q²  metrics",
    ],
    title_fs=9.5, item_fs=8.0,
)

arrow_down(FCX_R, y_botR2, y_botR2 - 0.30)
y_botR3 = node(
    FX_R, y_botR2 - 0.30, FW, C_STAT,
    "Feature Importance",
    [
        "vip_scores()  ·  VIP = f(w, t, q)",
        "S-plot  ·  covariance vs correlation",
        "UnivarStats  ·  t-test / Mann-Whitney / FDR",
    ],
    title_fs=9.5, item_fs=8.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# MERGE both branches → Interactive Visualization
# ─────────────────────────────────────────────────────────────────────────────
MERGE_Y = min(y_botL2, y_botR3) - 0.35

# left branch connector
ax.plot([FCX_L, FCX_L], [y_botL2, MERGE_Y + 0.05], color=C_ARROW, lw=2.2, zorder=5)
# right branch connector
ax.plot([FCX_R, FCX_R], [y_botR3, MERGE_Y + 0.05], color=C_ARROW, lw=2.2, zorder=5)
# horizontal merge line
ax.plot([FCX_L, FCX_R], [MERGE_Y + 0.05, MERGE_Y + 0.05],
        color=C_ARROW, lw=2.2, zorder=5)
# arrow down from merge
ax.annotate("", xy=(CX, MERGE_Y - 0.25), xytext=(CX, MERGE_Y + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                            lw=2.2, mutation_scale=14), zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# 6 INTERACTIVE VISUALIZATION & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
y_botViz = node(
    X0, MERGE_Y - 0.25, WW, C_VIZ,
    "6 Interactive Visualization & Output",
    [
        "Plotly  ·  scores plots / loadings / VIP / S-plot / trajectory / 3D PCA",
        "Dash apps  ·  STOCSY_app  ·  pickie_peak  (browser-based)",
        "pandas DataFrame export  ·  PNG / SVG / HTML figures",
    ],
    badge="plotting.py",
)

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C_INPUT, "Data Input"),
    (C_PRE,   "Preprocessing"),
    (C_NORM,  "Normalization"),
    (C_ALIGN, "Alignment"),
    (C_STAT,  "Statistical Modeling"),
    (C_VIZ,   "Visualization / Output"),
]

LEG_Y    = y_botViz - 0.55
LEG_SW   = 0.28    # swatch width
LEG_SH   = 0.22   # swatch height
n_items  = len(legend_items)
LEG_W    = WW
item_w   = LEG_W / n_items

for i, (col, label) in enumerate(legend_items):
    bx = X0 + i * item_w + (item_w - LEG_SW) / 2 - 0.50
    rbox(bx, LEG_Y, LEG_SW, LEG_SH, col, r=0.07, zorder=6)
    ax.text(bx + LEG_SW + 0.09, LEG_Y + LEG_SH / 2, label,
            ha="left", va="center", fontsize=7, color=DARK, zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
ax.text(CX, LEG_Y - 0.32,
        "Bubpamala T.  ·  kawa-technology  ·  github.com/aeiwz/metbit  "
        "(AI writing assistance: Claude Sonnet 4.6, GPT-5.2)",
        ha="center", va="center", fontsize=7.5,
        color=DARK, alpha=0.55, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE  – clip canvas to content
# ─────────────────────────────────────────────────────────────────────────────
out = "workflow_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close()
