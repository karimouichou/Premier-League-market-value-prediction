"""
Steps 5-6: modelling and evaluation for the PL market-value project.

Usage:  python 03_model.py pl_features.csv

Compares, with repeated stratified 5-fold CV (3 repeats), on log(market value):
  - naive baselines (position median; age+position+minutes only)
  - Ridge / RandomForest / GradientBoosting on the full feature set
  - ablations: without `rating`, without position interactions
  - one pooled model vs one model per position

Outputs: model_results.csv, model_misses.csv, model_comparison.png,
         model_pred_vs_actual.png, model_feature_importance.png
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "pl_features.csv"
N_REPEATS, N_SPLITS = 3, 5
ORDER = ["GK", "DEF", "MID", "FWD"]

df = pd.read_csv(SRC).reset_index(drop=True)
y = df["log_value"].values
pos = df["position"].values

NON_FEATURES = {"player_id", "player_name", "team_name", "position_stats", "position",
                "market_value_eur", "log_value"}
ALL = [c for c in df.select_dtypes("number").columns if c not in NON_FEATURES]
INTER = [c for c in ALL if "_x_" in c]
FEATURE_SETS = {
    "full": ALL,
    "no_rating": [c for c in ALL if c != "rating"],
    "no_interactions": [c for c in ALL if c not in INTER],
}
BASIC = ["age", "age_sq", "minutes", "pos_GK", "pos_DEF", "pos_MID", "pos_FWD"]


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.4,
                                     n_jobs=-1, random_state=0)
    if kind == "gbm":
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                         subsample=0.8, random_state=0)
    raise ValueError(kind)


def cv_predict(cols, kind, per_position=False):
    """Out-of-fold predictions, one array per repeat -> shape (N_REPEATS, n)."""
    out = np.zeros((N_REPEATS, len(df)))
    X = df[cols].values
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=r)
        for tr, te in skf.split(X, pos):
            if per_position:
                for p in ORDER:
                    tr_p, te_p = tr[pos[tr] == p], te[pos[te] == p]
                    if len(te_p) == 0:
                        continue
                    m = make_model(kind).fit(X[tr_p], y[tr_p])
                    out[r, te_p] = m.predict(X[te_p])
            else:
                m = make_model(kind).fit(X[tr], y[tr])
                out[r, te] = m.predict(X[te])
    return out


def baseline_position_median():
    out = np.zeros((N_REPEATS, len(df)))
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=r)
        for tr, te in skf.split(df, pos):
            med = pd.Series(y[tr]).groupby(pos[tr]).median()
            out[r, te] = pd.Series(pos[te]).map(med).values
    return out


def metrics(pred):
    """Mean +- std across repeats."""
    rows = []
    actual = df["market_value_eur"].values
    for r in range(pred.shape[0]):
        p = pred[r]
        eur = np.expm1(p)
        rows.append({
            "MAE_log": np.mean(np.abs(p - y)),
            "RMSE_log": np.sqrt(np.mean((p - y) ** 2)),
            "R2_log": 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
            "MAE_eur_M": np.mean(np.abs(eur - actual)) / 1e6,
            "MedAPE_pct": np.median(np.abs(eur - actual) / actual) * 100,
        })
    m = pd.DataFrame(rows)
    return m.mean().to_dict(), m.std().to_dict()


results, preds = [], {}


def record(name, pred, group):
    mean, std = metrics(pred)
    results.append({"model": name, "group": group, **{k: round(v, 3) for k, v in mean.items()},
                    "MAE_log_sd": round(std["MAE_log"], 3), "R2_log_sd": round(std["R2_log"], 3)})
    preds[name] = pred
    print(f"{name:42s} MAE_log {mean['MAE_log']:.3f}  R2 {mean['R2_log']:.3f}  "
          f"MAE €{mean['MAE_eur_M']:.1f}M  MedAPE {mean['MedAPE_pct']:.0f}%")


print(f"{len(df)} players, {len(ALL)} features, {N_REPEATS}x{N_SPLITS}-fold CV\n")
record("baseline: position median", baseline_position_median(), "baseline")
for kind in ("ridge", "gbm"):
    record(f"baseline: age+pos+minutes only [{kind}]", cv_predict(BASIC, kind), "baseline")

for fs_name, cols in FEATURE_SETS.items():
    for kind in ("ridge", "rf", "gbm"):
        record(f"{kind} [{fs_name}]", cv_predict(cols, kind), "pooled")

# Per-position models (interactions are redundant inside one position)
per_cols = [c for c in FEATURE_SETS["no_interactions"] if not c.startswith("pos_")]
for kind in ("ridge", "rf", "gbm"):
    record(f"{kind} [per-position]", cv_predict(per_cols, kind, per_position=True), "per-position")

res = pd.DataFrame(results)
res.to_csv("model_results.csv", index=False)

# ---------------------------------------------------------------------------
# Best pooled model -> per-position error, biggest misses
# ---------------------------------------------------------------------------
pooled = res[res.group == "pooled"].sort_values("MAE_log")
best_name = pooled.iloc[0]["model"]
print(f"\nBest pooled model by MAE_log: {best_name}")
best_pred = preds[best_name].mean(axis=0)  # average of the OOF predictions across repeats
df["pred_log"] = best_pred
df["pred_eur"] = np.expm1(best_pred)
df["err_log"] = best_pred - y

print("\nPer-position error, best pooled vs per-position version of the same model class:")
kind = best_name.split(" ")[0]
pp_name = f"{kind} [per-position]"
rows = []
for p in ORDER:
    m = pos == p
    for label, pr in (("pooled", preds[best_name].mean(0)), ("per-position", preds[pp_name].mean(0))):
        rows.append({"position": p, "approach": label, "n": int(m.sum()),
                     "MAE_log": np.mean(np.abs(pr[m] - y[m])),
                     "MAE_eur_M": np.mean(np.abs(np.expm1(pr[m]) - df.market_value_eur.values[m])) / 1e6})
pp = pd.DataFrame(rows).round(3)
print(pp.to_string(index=False))
pp.to_csv("model_per_position.csv", index=False)

cols_out = ["player_name", "team_name", "position", "age", "market_value_eur", "pred_eur"]
miss = df.assign(diff_log=df.err_log)[cols_out + ["diff_log"]]
miss["pred_eur"] = miss.pred_eur.round(-5)
under = miss.sort_values("diff_log", ascending=False).head(10).assign(verdict="model > TM value (TM undervalues vs stats)")
over = miss.sort_values("diff_log").head(10).assign(verdict="model < TM value (TM values on more than stats)")
pd.concat([under, over]).to_csv("model_misses.csv", index=False)
print("\nModel above Transfermarkt value:")
print(under[cols_out].to_string(index=False))
print("\nModel below Transfermarkt value:")
print(over[cols_out].to_string(index=False))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
euro = lambda x, _=None: f"€{x/1e6:.0f}M" if x >= 1e6 else f"€{x/1e3:.0f}k"

# 1. Model comparison (MAE in euros)
sel = res.sort_values("MAE_eur_M", ascending=False)
fig, ax = plt.subplots(figsize=(9, 0.38 * len(sel) + 1.4))
ax.barh(range(len(sel)), sel.MAE_eur_M, color=BLUE, height=0.62)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(sel.model, fontsize=9.5)
for i, v in enumerate(sel.MAE_eur_M):
    ax.text(v + 0.15, i, f"€{v:.1f}M", va="center", fontsize=9, color=INK2)
ax.set_xlabel("Mean absolute error, out-of-fold (€M)")
ax.set_title("Average miss in euros: every stats model beats the naive baselines")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=160)
plt.close(fig)

# 2. Predicted vs actual, small multiples by position
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
lo, hi = 1e5, 3e8
for ax, p in zip(axes, ORDER):
    m = pos == p
    ax.scatter(df.market_value_eur[m], df.pred_eur[m], s=22, color=BLUE, alpha=0.5, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"{p} (n={m.sum()})")
    ax.set_xlabel("Transfermarkt value")
    ax.set_xticks([1e6, 1e7, 1e8]); ax.set_yticks([1e6, 1e7, 1e8])
    ax.xaxis.set_major_formatter(FuncFormatter(euro)); ax.yaxis.set_major_formatter(FuncFormatter(euro))
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("Predicted value (out-of-fold)")
fig.suptitle(f"Predicted vs actual market value, {best_name}", x=0.01, ha="left",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.01, -0.03, "Each prediction comes from a model that never saw that player. Dashed line = perfect prediction.",
         color=INK2, fontsize=9.5)
fig.tight_layout()
fig.savefig("model_pred_vs_actual.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# 3. Permutation importance of the best pooled model, computed on held-out folds
cols = FEATURE_SETS[best_name.split("[")[1].rstrip("]")]
X = df[cols].values
imp = np.zeros(len(cols))
skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=0)
for tr, te in skf.split(X, pos):
    m = make_model(kind).fit(X[tr], y[tr])
    pi = permutation_importance(m, X[te], y[te], n_repeats=5, random_state=0,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    imp += pi.importances_mean / N_SPLITS
imp_s = pd.Series(imp, index=cols).sort_values(ascending=False)
imp_s.to_csv("model_feature_importance.csv", header=["mae_increase_log"], index_label="feature")
top = imp_s.head(15)[::-1]
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.barh(range(len(top)), top.values, color=BLUE, height=0.62)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([c.replace("_p90", "/90") for c in top.index], fontsize=9.5)
ax.set_xlabel("Increase in held-out MAE (log scale) when the feature is shuffled")
ax.set_title(f"What the model relies on ({best_name})")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("model_feature_importance.png", dpi=160)
plt.close(fig)
print("\nTop 10 features by permutation importance:")
print(imp_s.head(10).round(4).to_string())
print("\nDone.")
