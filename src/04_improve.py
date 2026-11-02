"""
Step 7: can we reduce the model misses?

Usage:  python 04_improve.py pl_features.csv

Compares, with the same repeated stratified CV as 03_model.py:
  A  baseline gradient boosting (full features)
  B  + team context features (teammates' average rating / goals / clean sheets, leave-one-out)
  C  B + team value level (mean value of teammates, computed INSIDE each fold)
  D  tuned gradient boosting on B
  E  blend of Ridge + tuned gradient boosting on B
  (XGBoost variants + SHAP run automatically if those packages are installed)

Outputs: improve_results.csv, improve_group_bias.csv, improve_results.png, shap_summary.png (if shap installed)
"""
import sys
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "pl_features.csv"
N_REPEATS, N_SPLITS = 3, 5

df = pd.read_csv(SRC).reset_index(drop=True)
y = df.log_value.values
pos = df.position.values
team = df.team_name.values

NON = {"player_id", "player_name", "team_name", "position_stats", "position", "market_value_eur", "log_value"}
BASE = [c for c in df.select_dtypes("number").columns if c not in NON]

# ---------------------------------------------------------------------------
# B. Team context from STATS only (no market values): leave-one-out over teammates
# ---------------------------------------------------------------------------
w = df.minutes.values
goals = df.goals_p90.values * w / 90
ctx = pd.DataFrame(index=df.index)
for name, val in {"team_rating": df.rating.values,
                  "team_cleansheet": df.cleanSheet_rate.values}.items():
    num = pd.Series(val * w).groupby(team).transform("sum") - val * w
    den = pd.Series(w).groupby(team).transform("sum") - w
    ctx[name] = num / den
ctx["team_goal_rate"] = ((pd.Series(goals).groupby(team).transform("sum") - goals) /
                         (pd.Series(w).groupby(team).transform("sum") - w) * 90)
CTX = list(ctx.columns)
df = pd.concat([df, ctx], axis=1)


def team_value_encoding(train_idx, apply_idx):
    """Mean log value of the player's TRAIN-fold teammates (excluding himself)."""
    s = pd.DataFrame({"team": team[train_idx], "v": y[train_idx]}, index=train_idx)
    tot, cnt = s.groupby("team").v.sum(), s.groupby("team").v.count()
    gmean = y[train_idx].mean()
    out = np.empty(len(apply_idx))
    train_set = set(train_idx)
    for k, i in enumerate(apply_idx):
        t = team[i]
        t_sum, t_cnt = tot.get(t, 0.0), cnt.get(t, 0)
        if i in train_set:  # leave himself out when he is a training row
            t_sum, t_cnt = t_sum - y[i], t_cnt - 1
        out[k] = t_sum / t_cnt if t_cnt > 0 else gmean
    return out


def gbm(**kw):
    p = dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=0)
    p.update(kw)
    return GradientBoostingRegressor(**p)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))


def run_cv(cols, model_fn, team_value=False, seeds=range(N_REPEATS)):
    out = np.zeros((len(list(seeds)), len(df)))
    for r, seed in enumerate(seeds):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for tr, te in skf.split(df, pos):
            Xtr, Xte = df.loc[tr, cols].copy(), df.loc[te, cols].copy()
            if team_value:
                Xtr["team_value_level"] = team_value_encoding(tr, tr)
                Xte["team_value_level"] = team_value_encoding(tr, te)
            m = model_fn() if callable(model_fn) else model_fn
            m.fit(Xtr.values, y[tr])
            out[r, te] = m.predict(Xte.values)
    return out


def score(pred):
    rows = []
    for p in pred:
        rows.append({
            "MAE_log": np.mean(np.abs(p - y)),
            "R2_log": 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
            "MAE_eur_M": np.mean(np.abs(np.expm1(p) - df.market_value_eur)) / 1e6,
            "MedAPE_pct": np.median(np.abs(np.expm1(p) - df.market_value_eur) / df.market_value_eur) * 100,
        })
    return pd.DataFrame(rows).mean().to_dict()


results, preds = [], {}


def record(name, pred):
    s = score(pred)
    results.append({"model": name, **{k: round(v, 3) for k, v in s.items()}})
    preds[name] = pred
    print(f"{name:52s} MAE_log {s['MAE_log']:.3f}  R2 {s['R2_log']:.3f}  "
          f"MAE €{s['MAE_eur_M']:.1f}M  MedAPE {s['MedAPE_pct']:.0f}%", flush=True)


record("A  baseline gbm", run_cv(BASE, gbm))
record("B  + team context (stats only)", run_cv(BASE + CTX, gbm))
record("C  B + teammates' value level (in-fold)", run_cv(BASE + CTX, gbm, team_value=True))

# D. Small hyper-parameter search. Seeds 10-12 are used for choosing so that the final
#    numbers (seeds 0-2) are not picked on the very same splits; some optimism remains.
grid = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_samples_leaf=leaf, loss=loss)
        for d, (lr, n), leaf, loss in itertools.product(
            (2, 3, 4), ((0.03, 300), (0.02, 500)), (3,), ("squared_error", "huber"))]
best_p, best_s = None, 9
for p in grid:
    s = score(run_cv(BASE + CTX, lambda p=p: gbm(**p), seeds=range(10, 12)))["MAE_log"]
    if s < best_s:
        best_p, best_s = p, s
print("\nbest hyper-parameters:", best_p, f"(MAE_log {best_s:.3f} on tuning splits)\n")
record("D  B + tuned gbm", run_cv(BASE + CTX, lambda: gbm(**best_p)))
pred_ridge = run_cv(BASE + CTX, ridge)
record("E  blend: 50% ridge + 50% tuned gbm", 0.5 * pred_ridge + 0.5 * preds["D  B + tuned gbm"])
record("   + teammates' value level, tuned gbm",
       run_cv(BASE + CTX, lambda: gbm(**best_p), team_value=True))

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    xgb = lambda: XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
                               colsample_bytree=0.6, reg_lambda=3.0, min_child_weight=2,
                               random_state=0, n_jobs=-1)
    record("XGBoost on B", run_cv(BASE + CTX, xgb))
except ImportError:
    print("(xgboost not installed -> skipped; `pip install xgboost` to include it)")

res = pd.DataFrame(results)
res.to_csv("improve_results.csv", index=False)

# ---------------------------------------------------------------------------
# Where did the errors go? signed log error (pred - actual) by group
# ---------------------------------------------------------------------------
groups = {
    "Burnley players": team == "Burnley",
    "Value >= €60M": df.market_value_eur.values >= 60e6,
    "Age >= 33": df.age.values >= 33,
    "Everyone else": ~((team == "Burnley") | (df.market_value_eur.values >= 60e6) | (df.age.values >= 33)),
}
rows = []
for name, pred in preds.items():
    p = pred.mean(0)
    for g, m in groups.items():
        rows.append({"model": name, "group": g, "n": int(m.sum()),
                     "mean_signed_err_log": round(float(np.mean(p[m] - y[m])), 3),
                     "MAE_log": round(float(np.mean(np.abs(p[m] - y[m]))), 3)})
gb = pd.DataFrame(rows)
gb.to_csv("improve_group_bias.csv", index=False)
print("\nSigned error (pred - actual, log scale) by group; 0 = unbiased, -0.5 ~ predicts 40% too low")
print(gb.pivot(index="model", columns="group", values="mean_signed_err_log").reindex(res.model).to_string())

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left"})
s = res.iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 0.55 * len(s) + 1.4))
ax.barh(range(len(s)), s.MAE_eur_M, color=BLUE, height=0.62)
ax.set_yticks(range(len(s))); ax.set_yticklabels(s.model, fontsize=9.5)
for i, v in enumerate(s.MAE_eur_M):
    ax.text(v + 0.1, i, f"€{v:.1f}M", va="center", fontsize=9, color=INK2)
ax.set_xlabel("Mean absolute error, out-of-fold (€M)")
ax.set_title("Reducing the average miss")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("improve_results.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------
# Optional: SHAP on the tuned model (fit on all data, for explanation only)
# ---------------------------------------------------------------------------
try:
    import shap
    cols = BASE + CTX
    model = gbm(**best_p).fit(df[cols].values, y)
    sv = shap.TreeExplainer(model).shap_values(df[cols])
    shap.summary_plot(sv, df[cols], show=False, max_display=15)
    plt.tight_layout(); plt.savefig("shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()
    print("saved shap_summary.png")
except ImportError:
    print("(shap not installed -> skipped; `pip install shap` to include it)")
