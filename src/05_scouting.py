"""
Step 8: "Undervalued players" scouting table.

Usage:  python 05_scouting.py predictions.csv

Idea: compare what the stats say a player is worth (out-of-fold model prediction) with what Transfermarkt
says. A raw gap is misleading, because the model "regresses to the mean": it over-predicts cheap players and
under-predicts stars, which would fill the list with 35-year-olds worth EUR 300k. So we:

  1. CALIBRATE the prediction: fit  log(actual) = a + b * log(predicted)  so the gap is unbiased across
     value levels (b > 1 undoes the compression of the tails).
  2. FILTER for scouting relevance (enough minutes, a value where percentages mean something, age cap for
     the "undervalued" list).
  3. Only call a gap "strong" when it exceeds the model's typical error.

Outputs: scouting_table.csv, scouting_undervalued_overvalued.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Run without arguments (e.g. VS Code "Run" button): work inside outputs/, next to the other results.
if len(sys.argv) < 2:
    import os as _os
    from pathlib import Path as _Path
    _os.chdir(_Path(__file__).resolve().parent.parent / "outputs")
SRC = sys.argv[1] if len(sys.argv) > 1 else "predictions.csv"
MIN_MINUTES = 900          # about 10 full matches
MIN_VALUE = 3_000_000      # below this, "x times higher" is meaningless
MAX_AGE_UNDERVALUED = 30   # market discounts age for resale/contract reasons; keep the list to buyable-age players
STRONG = 0.5               # log gap ~ model value 65% above (or 40% below) the market; ~1.5x typical error
N_LIST = 12

p = pd.read_csv(SRC)
y = np.log(p.market_value_eur)
x = np.log(p.pred_team_context_eur)

# 1. Calibration ------------------------------------------------------------
b, a = np.polyfit(x, y, 1)
p["model_value_eur"] = np.exp(a + b * x).round(-4).astype("int64")
p["gap_log"] = np.log(p.model_value_eur / p.market_value_eur)       # >0: stats say worth more than market
p["model_vs_market_x"] = (p.model_value_eur / p.market_value_eur).round(2)
p["gap_eur_M"] = ((p.model_value_eur - p.market_value_eur) / 1e6).round(1)
club_mean = p.groupby("team_name").gap_log.transform("mean")
p["gap_vs_club_log"] = (p.gap_log - club_mean).round(3)               # gap relative to the player's own club
p["gap_log"] = p.gap_log.round(3)

typical_err = np.mean(np.abs(p.gap_log))
print(f"calibration: log(actual) = {a:.3f} + {b:.3f} * log(pred)   (slope > 1 = tails were compressed)")
print(f"typical |gap| after calibration: {typical_err:.2f}  (strong-signal threshold: {STRONG})")

p["verdict"] = np.where(p.gap_log >= STRONG, "stats suggest higher value",
                np.where(p.gap_log <= -STRONG, "market pays a premium", "in line (within model error)"))

# 2. Eligibility ------------------------------------------------------------
ok = (p.minutes >= MIN_MINUTES) & (p.market_value_eur >= MIN_VALUE)
under = p[ok & (p.age <= MAX_AGE_UNDERVALUED)].sort_values("gap_log", ascending=False).head(N_LIST)
over = p[ok].sort_values("gap_log").head(N_LIST)

p.sort_values("gap_log", ascending=False).to_csv("scouting_table.csv", index=False)

show = ["player_name", "team_name", "position", "age", "market_value_eur", "model_value_eur", "gap_log",
        "gap_vs_club_log"]
fmt = lambda d: d[show].assign(market_value_eur=d.market_value_eur / 1e6,
                               model_value_eur=d.model_value_eur / 1e6).round(2).to_string(index=False)
print(f"\nStats suggest MORE than the market (age <= {MAX_AGE_UNDERVALUED}, >= {MIN_MINUTES} min, "
      f">= EUR {MIN_VALUE/1e6:.0f}M); values in EUR M:")
print(fmt(under))
print("\nMarket pays a PREMIUM over the stats (values in EUR M):")
print(fmt(over))
print("\nPlayers per verdict among eligible players:")
print(p[ok].verdict.value_counts().to_string())
print("Gap by group after calibration (mean log gap; should now be close to 0):")
print(p[ok].groupby(pd.cut(p[ok].market_value_eur, [0, 5e6, 15e6, 30e6, 60e6, 3e8]), observed=True)
      .gap_log.mean().round(2).to_string())

# 3. Chart: dumbbell, market value vs model value --------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
C_MARKET, C_MODEL = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
euro = lambda v, _=None: f"€{v/1e6:.0f}M"


def dumbbell(ax, d, title):
    d = d.iloc[::-1]
    ys = np.arange(len(d))
    for yi, (_, r) in zip(ys, d.iterrows()):
        ax.plot([r.market_value_eur, r.model_value_eur], [yi, yi], color=GRID, lw=2.5, zorder=1)
    ax.scatter(d.market_value_eur, ys, s=62, color=C_MARKET, marker="o", zorder=3, label="Transfermarkt value")
    ax.scatter(d.model_value_eur, ys, s=70, color=C_MODEL, marker="D", zorder=3, label="Stats-based model value")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.player_name} ({r.team_name}, {r.age})" for _, r in d.iterrows()], fontsize=9.5)
    ax.set_xscale("log")
    lo = min(d.market_value_eur.min(), d.model_value_eur.min()) * 0.8
    hi = max(d.market_value_eur.max(), d.model_value_eur.max()) * 1.25
    ax.set_xlim(lo, hi)
    ticks = [t for t in (3e6, 5e6, 10e6, 20e6, 30e6, 50e6, 100e6) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(euro))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title)


fig, axes = plt.subplots(2, 1, figsize=(9.5, 10.2))
dumbbell(axes[0], under, f"Stats suggest a higher value than the market (age ≤ {MAX_AGE_UNDERVALUED})")
dumbbell(axes[1], over, "Market pays a premium over what the stats explain")
fig.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=C_MARKET, markersize=8,
                           label="Transfermarkt value"),
                    Line2D([0], [0], marker="D", color="none", markerfacecolor=C_MODEL, markersize=8,
                           label="Stats-based model value")],
           loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.017))
fig.text(0.01, 0.005, "Out-of-fold, calibrated predictions. A gap smaller than the model's typical error is noise; "
                      "stats cannot see reputation, contracts or injuries.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("scouting_undervalued_overvalued.png", dpi=160)
plt.close(fig)
print("\nSaved scouting_table.csv and scouting_undervalued_overvalued.png")
