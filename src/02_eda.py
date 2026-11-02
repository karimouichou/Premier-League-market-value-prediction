"""
Step 4: EDA plots for the PL market-value project.

Usage:  python 02_eda.py pl_features.csv
Output: eda_age_vs_value.png, eda_value_by_position.png, eda_position_correlations.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SRC = sys.argv[1] if len(sys.argv) > 1 else "pl_features.csv"
df = pd.read_csv(SRC)

# Design tokens (light surface). One hue for magnitude/identity; text stays in ink colors.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE = "#2a78d6"
ORDER = ["GK", "DEF", "MID", "FWD"]
NAMES = {"GK": "Goalkeepers", "DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def euro(x, _=None):
    return f"€{x/1e6:.0f}M" if x >= 1e6 else f"€{x/1e3:.0f}k"


# ---------------------------------------------------------------------------
# 1. Age vs value, one panel per position (small multiples, shared axes)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
bins = [(16, 21), (22, 24), (25, 27), (28, 30), (31, 40)]
for ax, pos in zip(axes, ORDER):
    s = df[df.position == pos]
    ax.scatter(s.age, s.market_value_eur, s=22, color=BLUE, alpha=0.45, linewidths=0)
    xs, ys = [], []
    for lo, hi in bins:
        b = s[(s.age >= lo) & (s.age <= hi)]
        if len(b) >= 3:
            xs.append(b.age.mean())
            ys.append(b.market_value_eur.median())
    ax.plot(xs, ys, color=INK, lw=2, marker="o", ms=5, mfc=SURFACE, mec=INK, mew=1.5)
    ax.set_yscale("log")
    ax.set_title(f"{NAMES[pos]} (n={len(s)})")
    ax.set_xlabel("Age")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(euro))
    ax.set_yticks([1e6, 3e6, 1e7, 3e7, 1e8])
    ax.set_xlim(16, 40)
axes[0].set_ylabel("Market value (log scale)")
fig.suptitle("Value holds roughly flat until the late 20s, then drops sharply in every position",
             x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
fig.text(0.01, -0.03, "Dots = players (≥450 min). Line = median value per age band (16-21, 22-24, 25-27, 28-30, 31+).",
         color=INK2, fontsize=9.5)
fig.tight_layout()
fig.savefig("eda_age_vs_value.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Value by position (box + strip)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.2))
data = [df[df.position == p].market_value_eur.values for p in ORDER]
box_kw = dict(widths=0.5, patch_artist=True, showfliers=False,
              medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2),
              capprops=dict(color=INK2))
try:
    bp = ax.boxplot(data, orientation="horizontal", **box_kw)   # matplotlib >= 3.10
except TypeError:
    bp = ax.boxplot(data, vert=False, **box_kw)                 # older matplotlib
for patch in bp["boxes"]:
    patch.set(facecolor=BLUE, alpha=0.25, edgecolor=BLUE)
rng = np.random.default_rng(0)
for i, d in enumerate(data, start=1):
    ax.scatter(d, i + rng.uniform(-0.18, 0.18, len(d)), s=10, color=BLUE, alpha=0.5, linewidths=0)
ax.set_xscale("log")
ax.set_xticks([1e6, 3e6, 1e7, 3e7, 1e8])
ax.xaxis.set_major_formatter(FuncFormatter(euro))
ax.set_yticklabels([f"{NAMES[p]} (n={len(d)})" for p, d in zip(ORDER, data)])
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_title("Goalkeepers are valued lower (median €12M); outfield positions cluster at €22-25M")
fig.tight_layout()
fig.savefig("eda_value_by_position.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Which stats track value, per position (Spearman, top 7, no age/interaction columns)
# ---------------------------------------------------------------------------
skip = ("_x_", "pos_", "age", "market_value_eur", "log_value", "player_id", "position_stats")
cols = [c for c in df.select_dtypes("number").columns if not c.startswith(skip) and "_x_" not in c]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
for ax, pos in zip(axes, ORDER):
    s = df[df.position == pos]
    block = s[cols].loc[:, s[cols].nunique() > 1]
    corr = block.corrwith(s.log_value, method="spearman").dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(7)[::-1]
    ax.barh(range(len(top)), top.values, color=BLUE, height=0.62)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([c.replace("_p90", "/90") for c in top.index], fontsize=9.5)
    for i, v in enumerate(top.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-1.0, 0.9)  # room for value labels so they never touch the axis labels
    ax.set_title(f"{NAMES[pos]} (n={len(s)})")
    ax.set_xlabel("Spearman ρ with log value")
fig.suptitle("Stats most associated with market value, by position (age excluded)",
             x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("eda_position_correlations.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Printed summary for the README
print("Median value by position:")
print(df.groupby("position").market_value_eur.median().reindex(ORDER).map(euro).to_string())
print("\nMedian age-band value (all positions):")
for lo, hi in bins:
    b = df[(df.age >= lo) & (df.age <= hi)]
    print(f"  {lo}-{hi}: n={len(b):3d} median {euro(b.market_value_eur.median())}")
print("\nSaved 3 EDA plots")
