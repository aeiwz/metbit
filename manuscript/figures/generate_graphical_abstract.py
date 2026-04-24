"""
Graphical abstract for the metbit manuscript.
Output: graphical_abstract.png (1200 x 600 px, 150 dpi)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#F7F9FC"
DARK      = "#1A1A2E"
BLUE1     = "#1B4F72"   # deep navy  – header / stage 1
BLUE2     = "#2E86C1"   # mid blue   – stage 2
TEAL      = "#148F77"   # teal       – stage 3
PURPLE    = "#6C3483"   # purple     – stage 4
ORANGE    = "#CA6F1E"   # amber      – stage 5
ARROW     = "#566573"
WHITE     = "#FFFFFF"
LIGHT_BG  = "#EBF5FB"

fig = plt.figure(figsize=(16, 8), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")
ax.set_facecolor(BG)


# ── helpers ───────────────────────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, color, alpha=1.0, lw=0, radius=0.35):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=WHITE,
        facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    return box


def arrow(ax, x0, x1, y, color=ARROW, lw=2.5):
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=lw, mutation_scale=18,
        ), zorder=4,
    )


def stage_label(ax, x, y, title, items, color, title_size=11, item_size=8.5):
    ax.text(x, y, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=WHITE,
            zorder=5)
    for i, item in enumerate(items):
        ax.text(x, y - 0.52 - i * 0.42, item, ha="center", va="center",
                fontsize=item_size, color=WHITE, zorder=5)


def mini_nmr(ax, cx, cy, w=1.6, h=1.1, color=BLUE1):
    """Draw a tiny schematic NMR spectrum."""
    xs = np.linspace(cx - w/2, cx + w/2, 400)
    # several Lorentzian peaks
    peaks = [(cx - 0.5, 0.28, 0.04),
             (cx - 0.15, 0.40, 0.05),
             (cx + 0.1,  0.18, 0.03),
             (cx + 0.45, 0.32, 0.045)]
    ys = np.zeros_like(xs)
    for p0, amp, sig in peaks:
        ys += amp / (1 + ((xs - p0) / sig) ** 2)
    ys = cy + ys * (h * 0.55)
    baseline = cy + np.ones_like(xs) * 0.05

    ax.fill_between(xs, baseline, ys, alpha=0.35, color=color, zorder=6)
    ax.plot(xs, ys, color=color, lw=1.8, zorder=7)
    ax.plot([cx - w/2, cx + w/2], [cy + 0.05, cy + 0.05],
            color=color, lw=1, zorder=6, alpha=0.6)


def mini_scatter(ax, cx, cy, color1, color2, r=0.5):
    """Two-cluster scatter schematic."""
    rng = np.random.default_rng(42)
    for c, dx, dy in [(color1, -0.18, 0.08), (color2, 0.18, -0.08)]:
        xs = rng.normal(cx + dx, 0.11, 18)
        ys = rng.normal(cy + dy, 0.11, 18)
        ax.scatter(xs, ys, color=c, s=18, zorder=7, alpha=0.85,
                   edgecolors="white", linewidths=0.4)


def mini_bar(ax, cx, cy, color):
    """Tiny VIP bar chart."""
    vals = [0.62, 0.88, 0.50, 1.0, 0.72]
    bw = 0.11
    x_left  = cx - 0.38
    x_right = cx + 0.38
    for i, v in enumerate(vals):
        bx = cx - 0.32 + i * (bw + 0.04)
        ax.bar(bx, v * 0.55, width=bw, bottom=cy,
               color=color, alpha=0.85, zorder=7, linewidth=0)
    # threshold line — limited to bar chart x-range in data coordinates
    ax.plot([x_left, x_right], [cy + 0.35, cy + 0.35],
            color="red", lw=1.2, zorder=8, alpha=0.85)


def mini_heatmap(ax, cx, cy, w=1.0, h=0.6):
    """Tiny correlation heatmap schematic."""
    data = np.array([[1, 0.8, 0.3], [0.8, 1, 0.5], [0.3, 0.5, 1]])
    extent = [cx - w/2, cx + w/2, cy, cy + h]
    ax.imshow(data, extent=extent, cmap="coolwarm", vmin=-1, vmax=1,
              aspect="auto", zorder=7, origin="upper")


# ── title banner ──────────────────────────────────────────────────────────────
rounded_box(ax, 0.25, 6.95, 15.5, 0.9, BLUE1, radius=0.3)
ax.text(8, 7.4, "metbit  ·  End-to-End NMR Metabolomics Analysis in Python",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=WHITE, zorder=5)

# ── stage definitions  (x_left, width, colour, title, items) ─────────────────
stages = [
    (0.30,  2.60, BLUE1,   "1 Raw NMR Data",
     ["Bruker FID files", "1H NMR spectra", "nmrglue I/O"]),
    (3.35,  2.60, BLUE2,   "2 Preprocessing",
     ["Digital filter removal", "Zero-fill + FFT", "Auto phase correction",
      "Baseline correction", "Calibration (TSP/DSS)"]),
    (6.40,  2.60, TEAL,    "3 Normalization & Alignment",
     ["PQN normalization", "SNV / MSC", "icoshift alignment",
      "Peak detection"]),
    (9.45,  2.60, PURPLE,  "4 Statistical Modeling",
     ["PCA  (pareto / UV / MC)", "OPLS-DA + permutation test",
      "VIP scoring", "STOCSY analysis"]),
    (12.50, 3.00, ORANGE,  "5 Outputs",
     ["Interactive Plotly figures", "Dash web applications",
      "Biomarker candidates"]),
]

BOX_Y   = 1.15    # bottom of stage boxes
BOX_H   = 5.55    # height of stage boxes
ICON_Y  = 5.25    # vertical centre for icon area

for x0, w, col, title, items in stages:
    # main box
    rounded_box(ax, x0, BOX_Y, w, BOX_H, col, alpha=0.92, radius=0.35)
    # header strip
    rounded_box(ax, x0, BOX_Y + BOX_H - 0.72, w, 0.72, col, radius=0.3)
    ax.text(x0 + w/2, BOX_Y + BOX_H - 0.36, title,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=WHITE, zorder=5)
    # bullet items
    for i, item in enumerate(items):
        ax.text(x0 + w/2, BOX_Y + BOX_H - 1.15 - i * 0.56, f"• {item}",
                ha="center", va="center", fontsize=8.4, color=WHITE, zorder=5)

# ── arrows between stages ─────────────────────────────────────────────────────
arrow_y = BOX_Y + BOX_H / 2
for i in range(len(stages) - 1):
    x_start = stages[i][0] + stages[i][1] + 0.02
    x_end   = stages[i+1][0] - 0.02
    arrow(ax, x_start, x_end, arrow_y, color=ARROW, lw=3)

# ── schematic icons inside each box ──────────────────────────────────────────
# Stage 1: NMR spectrum icon
mini_nmr(ax, stages[0][0] + stages[0][1]/2, 2.35, w=2.0, h=1.0, color=WHITE)

# Stage 2: processed spectrum
mini_nmr(ax, stages[1][0] + stages[1][1]/2, 2.35, w=2.0, h=1.0, color=WHITE)
# add baseline hint
bx = np.linspace(stages[1][0]+0.3, stages[1][0]+2.3, 80)
ax.plot(bx, np.full_like(bx, 2.35), color=WHITE, lw=1.2,
        linestyle="--", alpha=0.5, zorder=8)

# Stage 3: aligned spectra stack
for k, dy in enumerate([0.4, 0, -0.4]):
    mini_nmr(ax, stages[2][0] + stages[2][1]/2, 2.55 + dy,
             w=1.8, h=0.55, color=WHITE)

# Stage 4: PCA scores scatter
mini_scatter(ax,
             stages[3][0] + stages[3][1]/2, 2.55,
             WHITE, "#D4E6F1")
# VIP bars below
mini_bar(ax, stages[3][0] + stages[3][1]/2, 1.55, WHITE)

# Stage 5: heatmap icon + text badge
mini_heatmap(ax, stages[4][0] + stages[4][1]/2, 2.60, w=1.8, h=0.90)
# small rounded badge: "Plotly · Dash"
bx = stages[4][0] + stages[4][1]/2
by = 1.90
badge_w, badge_h = 1.50, 0.38
rounded_box(ax, bx - badge_w/2, by - badge_h/2, badge_w, badge_h,
            WHITE, alpha=0.22, radius=0.16)
ax.text(bx, by, "Plotly  ·  Dash",
        ha="center", va="center", fontsize=8.5,
        color=WHITE, fontweight="bold", zorder=8)

# ── footer ────────────────────────────────────────────────────────────────────
ax.text(8, 0.62,
        "pip install metbit   ·   github.com/aeiwz/metbit   ·   MIT License",
        ha="center", va="center", fontsize=9, color=DARK, alpha=0.7, zorder=5)

ax.text(0.35, 0.62,
        "Bubpamala T. · kawa-technology · 2024  (AI-assisted writing: Claude Sonnet 4.6, GPT-5.2)",
        ha="left", va="center", fontsize=8, color=DARK, alpha=0.55, zorder=5)

# ── save ──────────────────────────────────────────────────────────────────────
out = "graphical_abstract.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.close()
