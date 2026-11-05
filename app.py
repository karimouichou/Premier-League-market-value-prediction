"""Premier League market-value app (Streamlit).

    streamlit run app.py

Pages: Overview, Predict a value, Model vs market, Scouting table, Insights.
Everything is read from outputs/ (produced by run_all.py); the gradient-boosting model is
re-fitted on start-up (a few seconds, cached) so the Predict page can score new stat lines.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# EA-FC-style palette: deep navy, neon green / cyan accents, gold for the player card
BG, PANEL, PANEL2, LINE = "#080c18", "#111a2f", "#16213b", "#24325a"
ACCENT, ACCENT2, GOLD = "#00ff87", "#3ee6ff", "#f5c542"
INK, MUTED = "#f2f5ff", "#8b95b0"
GREY = MUTED
POS_COLORS = {"GK": "#ffb02e", "DEF": "#3ea6ff", "MID": "#00ff87", "FWD": "#ff4d8d"}
POSITIONS = ["GK", "DEF", "MID", "FWD"]
CTX = ["team_rating", "team_cleansheet", "team_goal_rate"]
NON_FEATURES = {"player_id", "player_name", "team_name", "position_stats", "position",
                "market_value_eur", "log_value"}
FONT = "Barlow, 'Segoe UI', sans-serif"

# Same position x stat interactions as src/01_clean_and_features.py
KEY_STATS = {
    "FWD": ["goals_p90", "expectedGoals_p90", "shotsOnTarget_p90", "goal_contrib_p90", "bigChancesMissed_p90"],
    "MID": ["ballRecovery_p90", "keyPasses_p90", "expectedAssists_p90", "chance_creation_p90",
            "accurateFinalThirdPasses_p90"],
    "DEF": ["cleanSheet_rate", "ballRecovery_p90", "interceptions_p90", "clearances_p90", "def_actions_p90"],
    "GK": ["cleanSheet_rate", "saves_p90", "goalsPrevented_p90"],
}

# Stats the user can edit on the Predict page: (column, label)
SLIDERS = {
    "FWD": [("goals_p90", "Goals per 90"), ("expectedGoals_p90", "xG per 90"), ("assists_p90", "Assists per 90"),
            ("shotsOnTarget_p90", "Shots on target per 90"), ("successfulDribbles_p90", "Dribbles won per 90"),
            ("keyPasses_p90", "Key passes per 90"), ("bigChancesMissed_p90", "Big chances missed per 90")],
    "MID": [("keyPasses_p90", "Key passes per 90"), ("expectedAssists_p90", "xA per 90"),
            ("ballRecovery_p90", "Ball recoveries per 90"), ("successfulDribbles_p90", "Dribbles won per 90"),
            ("tackles_p90", "Tackles per 90"), ("goals_p90", "Goals per 90"),
            ("accuratePassesPercentage", "Pass accuracy (%)")],
    "DEF": [("cleanSheet_rate", "Clean-sheet rate"), ("interceptions_p90", "Interceptions per 90"),
            ("clearances_p90", "Clearances per 90"), ("ballRecovery_p90", "Ball recoveries per 90"),
            ("aerialDuelsWon_p90", "Aerial duels won per 90"), ("successfulDribbles_p90", "Dribbles won per 90"),
            ("accuratePassesPercentage", "Pass accuracy (%)")],
    "GK": [("cleanSheet_rate", "Clean-sheet rate"), ("saves_p90", "Saves per 90"),
           ("goalsPrevented_p90", "Goals prevented per 90"), ("highClaims_p90", "High claims per 90"),
           ("accuratePassesPercentage", "Pass accuracy (%)")],
}

st.set_page_config(page_title="PL Market Value", page_icon="⚽", layout="wide")

# Plotly: one dark template for every chart
_tpl = go.layout.Template(pio.templates["plotly_dark"])
_tpl.layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=INK, size=14),
    colorway=[ACCENT, ACCENT2, GOLD, "#ff4d8d", "#3ea6ff", "#ffb02e"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
    title=dict(font=dict(family="Barlow Condensed, sans-serif", size=22)),
    hoverlabel=dict(bgcolor=PANEL2, font_family=FONT),
)
pio.templates["fc"] = _tpl
pio.templates.default = "fc"

st.markdown(
    f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Barlow:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stMarkdown, p, label, li {{font-family: {FONT};}}
    .stApp {{background: radial-gradient(1200px 600px at 85% -10%, #12306a55 0%, transparent 60%),
                         radial-gradient(900px 500px at -10% 10%, #00ff8718 0%, transparent 55%), {BG};}}
    .block-container {{padding-top: 2.2rem; max-width: 1250px;}}
    h1, h2, h3, h4 {{font-family: 'Barlow Condensed', sans-serif !important; text-transform: uppercase;
                     letter-spacing: .04em; color: {INK}; font-weight: 800;}}
    h1 {{font-size: 3rem !important; line-height: 1;
         background: linear-gradient(90deg, {INK} 30%, {ACCENT} 100%); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent;}}
    h2, h3 {{border-left: 4px solid {ACCENT}; padding-left: .6rem;}}
    [data-testid="stCaptionContainer"] {{color: {MUTED};}}
    /* metric cards */
    [data-testid="stMetric"] {{background: linear-gradient(160deg, {PANEL2}, {PANEL}); border: 1px solid {LINE};
        border-radius: 14px; padding: 14px 18px; box-shadow: 0 6px 24px #0007;}}
    [data-testid="stMetricLabel"] p {{text-transform: uppercase; letter-spacing: .08em; color: {MUTED};
        font-weight: 600; font-size: .8rem;}}
    [data-testid="stMetricValue"] {{font-family: 'Barlow Condensed', sans-serif; font-weight: 800;
        font-size: 2.4rem; color: {ACCENT};}}
    /* sidebar */
    [data-testid="stSidebar"] {{background: linear-gradient(180deg, #0c1530, {BG}); border-right: 1px solid {LINE};}}
    [data-testid="stSidebar"] [role="radiogroup"] label {{padding: 6px 10px; border-radius: 8px;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{background: {PANEL2};}}
    /* tabs */
    button[data-baseweb="tab"] {{font-family: 'Barlow Condensed', sans-serif; font-size: 1.15rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: .06em;}}
    /* tables / images / alerts */
    [data-testid="stDataFrame"], [data-testid="stImage"] img, [data-testid="stPlotlyChart"] {{border-radius: 12px;}}
    [data-testid="stImage"] img {{border: 1px solid {LINE};}}
    [data-testid="stAlert"] {{border-radius: 12px; border: 1px solid {LINE};}}
    hr {{border-color: {LINE};}}
    /* FUT-style player card */
    .fut {{position: relative; border-radius: 18px 18px 40px 40px; padding: 22px 22px 26px; text-align: center;
        background: linear-gradient(155deg, #2b2410 0%, #4a3a12 35%, #1a1608 100%); border: 2px solid {GOLD};
        box-shadow: 0 0 32px #f5c54233, inset 0 0 40px #f5c54222; margin-bottom: 14px;}}
    .fut .val {{font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 4.2rem; line-height: 1;
        color: {GOLD}; text-shadow: 0 0 18px #f5c54255;}}
    .fut .pos {{display: inline-block; font-family: 'Barlow Condensed'; font-weight: 800; font-size: 1.3rem;
        letter-spacing: .12em; color: #0b0b0b; padding: 2px 14px; border-radius: 6px; margin: 4px 0 10px;}}
    .fut .name {{font-family: 'Barlow Condensed'; font-weight: 800; font-size: 1.7rem; text-transform: uppercase;
        letter-spacing: .06em; color: {INK}; border-top: 1px solid #f5c54255; border-bottom: 1px solid #f5c54255;
        padding: 6px 0; margin-bottom: 8px;}}
    .fut .sub {{color: #d9cfa8; font-size: .95rem; letter-spacing: .04em;}}
    .fut .range {{margin-top: 10px; color: {ACCENT}; font-weight: 600; font-size: .95rem;}}
    </style>""",
    unsafe_allow_html=True,
)


def player_card(name, sub, pos, value, rng):
    st.markdown(
        f"""<div class="fut"><div class="val">{value}</div>
        <div class="pos" style="background:{POS_COLORS[pos]}">{pos}</div>
        <div class="name">{name}</div><div class="sub">{sub}</div>
        <div class="range">80% range · {rng}</div></div>""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def eur(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"€{x / 1e6:,.1f}M"
    return f"€{x / 1e3:,.0f}k"


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None)


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(OUT / "pl_features.csv").reset_index(drop=True)
    base_cols = [c for c in df.select_dtypes("number").columns if c not in NON_FEATURES]

    # Team context (same definition as src/04_improve.py): minutes-weighted stats of teammates
    w = df.minutes.values.astype(float)
    goals = df.goals_p90.values * w / 90
    team = df.team_name
    loo = pd.DataFrame(index=df.index)
    for name, val in {"team_rating": df.rating.values, "team_cleansheet": df.cleanSheet_rate.values}.items():
        num = pd.Series(val * w).groupby(team.values).transform("sum") - val * w
        den = pd.Series(w).groupby(team.values).transform("sum") - w
        loo[name] = num / den
    g_sum = pd.Series(goals).groupby(team.values).transform("sum") - goals
    w_sum = pd.Series(w).groupby(team.values).transform("sum") - w
    loo["team_goal_rate"] = g_sum / w_sum * 90

    t = pd.DataFrame({"w": w, "rw": df.rating * w, "cw": df.cleanSheet_rate * w, "g": goals, "team": team})
    s = t.groupby("team").sum()
    team_ctx = pd.DataFrame({"team_rating": s.rw / s.w, "team_cleansheet": s.cw / s.w,
                             "team_goal_rate": s.g / s.w * 90})
    full = pd.concat([df, loo], axis=1)
    preds = pd.read_csv(OUT / "predictions.csv")
    full = full.merge(preds[["player_name", "team_name", "pred_team_context_eur", "pred_baseline_eur"]],
                      on=["player_name", "team_name"], how="left")
    return full, base_cols, team_ctx


@st.cache_resource(show_spinner="Training the model (a few seconds, cached)...")
def load_models():
    df, base_cols, _ = load_data()
    feats = base_cols + CTX
    models = [GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                                        random_state=s).fit(df[feats].values, df.log_value.values)
              for s in range(5)]  # 5 seeds averaged: subsampling makes a single fit slightly noisy
    return models, feats


def predict_log(rows: pd.DataFrame) -> np.ndarray:
    models, feats = load_models()
    X = rows[feats].values
    return np.mean([m.predict(X) for m in models], axis=0)


def derive(r: dict) -> dict:
    """Recompute every engineered feature that depends on the editable inputs."""
    r = dict(r)
    r["age_sq"] = r["age"] ** 2
    for p in POSITIONS:
        r[f"pos_{p}"] = int(r["position"] == p)
    r["goal_contrib_p90"] = r["goals_p90"] + r["assists_p90"]
    r["xG_xA_p90"] = r["expectedGoals_p90"] + r["expectedAssists_p90"]
    r["def_actions_p90"] = (r["tackles_p90"] + r["interceptions_p90"] + r["clearances_p90"]
                            + r["ballRecovery_p90"])
    r["chance_creation_p90"] = r["keyPasses_p90"] + r["bigChancesCreated_p90"]
    r["xG_overperf_p90"] = r["goals_p90"] - r["expectedGoals_p90"]
    for pos, cols in KEY_STATS.items():
        for c in cols:
            r[f"{pos}_x_{c}"] = r[f"pos_{pos}"] * r[c]
    return r


@st.cache_data
def residual_quantiles():
    p = pd.read_csv(OUT / "predictions.csv")
    r = np.log(p.market_value_eur / p.pred_team_context_eur)
    return float(r.quantile(0.10)), float(r.quantile(0.90))


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_overview():
    st.title("⚽ Premier League market value predictor")
    st.caption("2025/26 season · 383 players with 450+ minutes · target: Transfermarkt market value")
    st.write(
        "Can a player's season statistics, age and position explain his Transfermarkt market value? "
        "This app lets you explore the model, test what-if scenarios and browse players the market values "
        "differently from what the stats suggest."
    )

    imp = pd.read_csv(OUT / "improve_results.csv")
    base = imp[imp.model.str.startswith("A")].iloc[0]
    best = imp[imp.model.str.startswith("B")].iloc[0]
    c = st.columns(4)
    c[0].metric("Players", "383")
    c[1].metric("R² (log value)", f"{best.R2_log:.2f}", f"{best.R2_log - base.R2_log:+.2f} vs stats only")
    c[2].metric("Average miss", f"€{best.MAE_eur_M:.1f}M", f"{best.MAE_eur_M - base.MAE_eur_M:+.1f}M", delta_color="inverse")
    c[3].metric("Median error", f"{best.MedAPE_pct:.0f}%", f"{best.MedAPE_pct - base.MedAPE_pct:+.0f} pts", delta_color="inverse")
    st.caption("Repeated stratified 5-fold cross-validation (3 repeats): every prediction comes from a model "
               "that never saw that player. Deltas compare team-context model to the baseline gradient boosting.")

    left, right = st.columns([3, 2])
    with left:
        res = pd.read_csv(OUT / "model_results.csv")
        res = res[res.group.isin(["baseline", "pooled"]) & ~res.model.str.contains("no_")]
        extra = imp[imp.model.str.contains("team context")].iloc[[0]]
        rows = pd.concat([
            res[["model", "MAE_eur_M"]],
            pd.DataFrame({"model": ["gbm + team context"], "MAE_eur_M": extra.MAE_eur_M.values}),
        ])
        rows = rows.sort_values("MAE_eur_M", ascending=False)
        fig = px.bar(rows, x="MAE_eur_M", y="model", orientation="h", text=rows.MAE_eur_M.map(lambda v: f"€{v:.1f}M"),
                     labels={"MAE_eur_M": "Average miss (€M, out-of-fold)", "model": ""})
        fig.update_traces(marker_color=ACCENT, textposition="outside", textfont_color=INK)
        fig.update_layout(height=380, margin=dict(l=0, r=30, t=10, b=0))
        show(fig)
    with right:
        st.subheader("Key takeaways")
        st.markdown(
            "- **Age explains most of the value.** Age + position + minutes alone reach R² 0.67.\n"
            "- **Team context was the lever that worked.** Teammates' average rating, goal rate and "
            "clean sheets cut the average miss by about €0.9M.\n"
            "- **Stars are under-predicted** (Haaland: €200M listed vs. well below that predicted); stats can't see "
            "reputation.\n"
            "- **Tuning and blending did not help**: the information in the features is the bottleneck."
        )
    st.info("Use the sidebar to switch pages. **Predict a value** is the interactive part.")


def page_predict():
    st.title("Predict a value")
    st.write("Start from a real player or a typical player at a position, move the sliders, and see how the "
             "model's estimate changes.")
    df, _, team_ctx = load_data()
    lo_q, hi_q = residual_quantiles()

    mode = st.radio("Start from", ["A real player", "A typical player"], horizontal=True)
    if mode == "A real player":
        labels = (df.player_name + " (" + df.team_name + ")").tolist()
        pick = st.selectbox("Player", labels, index=labels.index("Erling Haaland (Manchester City)")
                            if "Erling Haaland (Manchester City)" in labels else 0)
        idx = labels.index(pick)
        player = df.loc[idx]
        pos, key = player.position, f"p{idx}"
        base_row = player.to_dict()
        pool = df[df.position == pos]
    else:
        pos = st.selectbox("Position", POSITIONS, index=3)
        pool = df[df.position == pos]
        base_row = pool.median(numeric_only=True).to_dict()
        base_row.update(position=pos)
        key, idx = f"t{pos}", None

    ctrl, res = st.columns([3, 2], gap="large")
    with ctrl:
        st.subheader("Profile")
        c1, c2 = st.columns(2)
        age = c1.slider("Age", 17, 40, int(base_row["age"]), key=f"{key}_age")
        minutes = c2.slider("Minutes played", 450, 3420, int(min(max(base_row["minutes"], 450), 3420)),
                            step=30, key=f"{key}_min")
        c3, c4 = st.columns(2)
        start_ratio = c3.slider("Share of games started", 0.0, 1.0, float(base_row["start_ratio"]),
                                step=0.05, key=f"{key}_sr")
        r_lo, r_hi = float(pool.rating.quantile(0.02)), float(pool.rating.quantile(0.98))
        rating = c4.slider("Average match rating", round(min(r_lo, base_row["rating"]), 2),
                           round(max(r_hi, base_row["rating"]), 2), float(round(base_row["rating"], 2)),
                           step=0.05, key=f"{key}_rat")

        st.subheader(f"Key {pos} stats")
        vals = {}
        cols = st.columns(2)
        for i, (col, label) in enumerate(SLIDERS[pos]):
            default = float(base_row[col])
            lo = min(float(pool[col].quantile(0.02)), default)
            hi = max(float(pool[col].quantile(0.98)), default)
            if hi - lo < 1e-6:
                hi = lo + 0.1
            step = max(round((hi - lo) / 100, 3), 0.001)
            vals[col] = cols[i % 2].slider(label, float(lo), float(hi), default, step=step, key=f"{key}_{col}")

        st.subheader("Club")
        own_team = base_row.get("team_name")
        teams = sorted(team_ctx.index)
        opts = ["League average"] + teams if own_team is None else teams
        team_choice = st.selectbox("Team context", opts,
                                   index=0 if own_team is None else teams.index(own_team), key=f"{key}_team",
                                   help="Teammates' average rating, goal rate and clean-sheet rate feed the model.")

    # Build the modified feature row
    row = dict(base_row)
    row.update(age=age, minutes=minutes, start_ratio=start_ratio, rating=rating, **vals)
    row["appearances"] = max(1.0, base_row["appearances"] * minutes / max(base_row["minutes"], 1))
    row = derive(row)

    def ctx_for(team_name):
        if team_name == "League average":
            return team_ctx.mean().to_dict()
        if idx is not None and team_name == own_team:
            return {c: base_row[c] for c in CTX}  # leave-one-out context, as in validation
        return team_ctx.loc[team_name].to_dict()

    def estimate(r, team_name):
        r = {**r, **ctx_for(team_name)}
        z = predict_log(pd.DataFrame([r]))[0]
        if idx is None:
            return float(np.expm1(z))
        # Real player: anchor on the out-of-fold estimate (the model never saw him) and apply the model's
        # change between his original profile and the edited one.
        orig = {**derive(base_row), **{c: base_row[c] for c in CTX}}
        z0 = predict_log(pd.DataFrame([orig]))[0]
        return float(base_row["pred_team_context_eur"] * np.exp(z - z0))

    est = estimate(row, team_choice)
    low, high = est * np.exp(lo_q), est * np.exp(hi_q)

    with res:
        st.subheader("Estimated value")
        if idx is not None:
            card_name, card_sub = base_row["player_name"], f"{base_row['team_name']} · age {age}"
        else:
            card_name, card_sub = f"Typical {pos}", f"age {age} · {team_choice}"
        player_card(card_name, card_sub, pos, eur(est), f"{eur(low)} – {eur(high)}")
        if idx is not None:
            tm = base_row["market_value_eur"]
            st.metric("Transfermarkt value", eur(tm), f"{(est / tm - 1) * 100:+.0f}% model vs. market",
                      delta_color="off")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[tm], y=[""], orientation="h", marker_color=MUTED, name="Transfermarkt"))
            fig.add_trace(go.Bar(x=[est], y=[" "], orientation="h", marker_color=ACCENT, name="Model"))
            fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), showlegend=True, barmode="group",
                              xaxis_title="€")
            show(fig)

        # Value-by-age curve for this stat line
        ages = list(range(18, 39))
        curve = [estimate(derive({**row, "age": a}), team_choice) / 1e6 for a in ages]
        fig = go.Figure(go.Scatter(x=ages, y=curve, mode="lines", line=dict(color=ACCENT, width=4),
                                   name="Model value"))
        fig.add_trace(go.Scatter(x=[age], y=[est / 1e6], mode="markers", marker=dict(size=13, color=GOLD),
                                 name="Current age"))
        fig.update_layout(title="Same stat line at different ages", height=300, margin=dict(l=0, r=0, t=40, b=0),
                          xaxis_title="Age", yaxis_title="Value (€M)", showlegend=False)
        show(fig)
        st.caption(
            "The range comes from out-of-fold errors (10th–90th percentile of real/predicted). "
            "Treat the estimate as 'what similar stats and age usually earn', not as a price. "
            "Stars and very old players are typically under-predicted."
        )


def page_explorer():
    st.title("Model vs. market")
    st.write("Each dot is a player. On the diagonal, the model agrees with Transfermarkt. "
             "Predictions are out-of-fold: the model never saw the player it is scoring.")
    df, _, _ = load_data()
    f1, f2 = st.columns(2)
    pos_sel = f1.multiselect("Position", POSITIONS, default=POSITIONS)
    team_sel = f2.multiselect("Team", sorted(df.team_name.unique()), placeholder="All teams")
    d = df[df.position.isin(pos_sel)]
    if team_sel:
        d = d[d.team_name.isin(team_sel)]
    d = d.assign(actual_M=d.market_value_eur / 1e6, pred_M=d.pred_team_context_eur / 1e6)

    fig = px.scatter(d, x="actual_M", y="pred_M", color="position", color_discrete_map=POS_COLORS,
                     hover_name="player_name", hover_data={"team_name": True, "age": True, "actual_M": ":.1f",
                                                           "pred_M": ":.1f", "position": False},
                     log_x=True, log_y=True,
                     labels={"actual_M": "Transfermarkt value (€M)", "pred_M": "Model value (€M)"})
    lim = [0.1, 250]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=MUTED, dash="dash"), showlegend=False,
                             hoverinfo="skip"))
    hl = st.selectbox("Highlight a player", [""] + sorted(d.player_name.tolist()))
    if hl:
        p = d[d.player_name == hl].iloc[0]
        fig.add_trace(go.Scatter(x=[p.actual_M], y=[p.pred_M], mode="markers+text", text=[hl],
                                 textposition="top center", marker=dict(size=16, color="rgba(0,0,0,0)",
                                                                        line=dict(color=GOLD, width=3)),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    show(fig)

    if d.empty:
        return
    st.subheader("Biggest misses")
    t = d.assign(error_pct=(d.pred_team_context_eur / d.market_value_eur - 1) * 100)
    table = t[["player_name", "team_name", "position", "age", "actual_M", "pred_M", "error_pct"]]
    cfg = {"player_name": "Player", "team_name": "Team", "position": "Pos", "age": "Age",
           "actual_M": st.column_config.NumberColumn("Market (€M)", format="%.1f"),
           "pred_M": st.column_config.NumberColumn("Model (€M)", format="%.1f"),
           "error_pct": st.column_config.NumberColumn("Model vs market", format="%+.0f%%")}
    a, b = st.columns(2)
    a.markdown("**Model above market**")
    a.dataframe(table.sort_values("error_pct", ascending=False).head(10), hide_index=True, column_config=cfg)
    b.markdown("**Model below market**")
    b.dataframe(table.sort_values("error_pct").head(10), hide_index=True, column_config=cfg)


def page_scouting():
    st.title("Scouting table")
    st.write("Players whose Transfermarkt value differs from what their stats suggest. "
             "The gap is calibrated so it is not biased by value level (see README).")
    st.warning("Not validated: there are no future values to test whether flagged players later gained value. "
               "Read 'stats suggest higher value' as 'worth a look', not 'bargain'.")
    s = pd.read_csv(OUT / "scouting_table.csv")

    with st.sidebar:
        st.markdown("### Scouting filters")
        pos = st.multiselect("Position", POSITIONS, default=POSITIONS, key="s_pos")
        teams = st.multiselect("Team", sorted(s.team_name.unique()), placeholder="All teams", key="s_team")
        min_min = st.slider("Minimum minutes", 450, 2500, 900, step=50)
        min_val = st.select_slider("Minimum market value", [0.15, 1, 3, 5, 10, 20], value=3,
                                   format_func=lambda v: f"€{v:g}M")
        thr = st.slider("Strong-gap threshold (log)", 0.2, 1.0, 0.5, step=0.05,
                        help="0.5 is about 1.5x the model's typical error.")

    d = s[s.position.isin(pos) & (s.minutes >= min_min) & (s.market_value_eur >= min_val * 1e6)]
    if teams:
        d = d[d.team_name.isin(teams)]
    d = d.assign(
        verdict=np.where(d.gap_log >= thr, "stats suggest higher value",
                         np.where(d.gap_log <= -thr, "market pays a premium", "in line")),
        market_M=d.market_value_eur / 1e6, model_M=d.model_value_eur / 1e6,
    )
    c = st.columns(3)
    c[0].metric("Eligible players", len(d))
    c[1].metric("Stats suggest higher value", int((d.verdict == "stats suggest higher value").sum()))
    c[2].metric("Market pays a premium", int((d.verdict == "market pays a premium").sum()))

    cfg = {"player_name": "Player", "team_name": "Team", "position": "Pos", "age": "Age", "minutes": "Minutes",
           "market_M": st.column_config.NumberColumn("Market (€M)", format="%.1f"),
           "model_M": st.column_config.NumberColumn("Model (€M)", format="%.1f"),
           "gap_log": st.column_config.NumberColumn("Gap (log)", format="%+.2f"),
           "gap_vs_club_log": st.column_config.NumberColumn("Gap vs club", format="%+.2f",
                                                            help="Gap relative to the player's own club: separates player-level from club-level signals."),
           "verdict": "Verdict"}
    cols = ["player_name", "team_name", "position", "age", "minutes", "market_M", "model_M", "gap_log",
            "gap_vs_club_log", "verdict"]

    def dumbbell(t, title):
        t = t.iloc[::-1]
        fig = go.Figure()
        for _, r in t.iterrows():
            fig.add_trace(go.Scatter(x=[r.market_M, r.model_M], y=[r.player_name] * 2, mode="lines",
                                     line=dict(color=LINE, width=4), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=t.market_M, y=t.player_name, mode="markers", name="Transfermarkt",
                                 marker=dict(color=GREY, size=11)))
        fig.add_trace(go.Scatter(x=t.model_M, y=t.player_name, mode="markers", name="Model (stats)",
                                 marker=dict(color=ACCENT, size=12)))
        fig.update_layout(title=title, height=max(320, 28 * len(t) + 120), xaxis_title="Value (€M)",
                          margin=dict(l=0, r=0, t=50, b=0), legend=dict(orientation="h", y=-0.12))
        show(fig)

    t1, t2, t3 = st.tabs(["Stats suggest higher value", "Market pays a premium", "Full table"])
    with t1:
        max_age = st.slider("Maximum age", 18, 40, 30, help="Older players are discounted for resale/contract "
                                                           "reasons; 30 keeps the list to buyable-age players.")
        u = d[(d.gap_log >= thr) & (d.age <= max_age)].sort_values("gap_log", ascending=False)
        if u.empty:
            st.info("No players match these filters.")
        else:
            dumbbell(u.head(15), "Largest gaps: model above market")
            st.dataframe(u[cols], hide_index=True, column_config=cfg)
    with t2:
        o = d[d.gap_log <= -thr].sort_values("gap_log")
        if o.empty:
            st.info("No players match these filters.")
        else:
            dumbbell(o.head(15), "Largest gaps: market above model")
            st.dataframe(o[cols], hide_index=True, column_config=cfg)
            st.caption("Mostly reputation: stars, goalkeepers in their 30s at big clubs, and young players "
                       "valued on potential.")
    with t3:
        st.dataframe(d.sort_values("gap_log", ascending=False)[cols], hide_index=True, column_config=cfg)
        st.download_button("Download filtered table (CSV)", d[cols].to_csv(index=False).encode("utf-8"),
                           "scouting_filtered.csv", "text/csv")


def page_insights():
    st.title("Insights")
    df, _, _ = load_data()
    t1, t2, t3, t4 = st.tabs(["What drives value", "Age & position", "Where it goes wrong", "Limitations"])

    with t1:
        imp = pd.read_csv(OUT / "model_feature_importance.csv").sort_values("mae_increase_log", ascending=False)
        top = imp.head(15).iloc[::-1]
        fig = px.bar(top, x="mae_increase_log", y="feature", orientation="h",
                     labels={"mae_increase_log": "Increase in error when shuffled (log MAE)", "feature": ""},
                     title="Permutation importance (held-out folds, baseline model)")
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=50, b=0))
        show(fig)
        st.markdown("**SHAP summary** (team-context model): right = pushes value up, left = down (log scale).")
        st.image(str(OUT / "shap_summary.png"))
        st.caption("SHAP shows what the model relies on, not what causes value: team features may partly stand "
                   "for club prestige.")

    with t2:
        pos = st.multiselect("Position", POSITIONS, default=POSITIONS, key="i_pos")
        d = df[df.position.isin(pos)]
        fig = px.scatter(d, x="age", y="market_value_eur", color="position", color_discrete_map=POS_COLORS,
                         hover_name="player_name", log_y=True, hover_data=["team_name"],
                         labels={"market_value_eur": "Market value (€, log)", "age": "Age"})
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
        show(fig)
        fig = px.box(df, x="position", y="market_value_eur", color="position", color_discrete_map=POS_COLORS,
                     category_orders={"position": POSITIONS}, log_y=True,
                     labels={"market_value_eur": "Market value (€, log)", "position": ""})
        fig.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        show(fig)
        st.markdown("**Hypothesis check:** clean-sheet rate is a real signal for goalkeepers (Spearman +0.65 with "
                    "value). Ball recoveries are *not* a top driver for defenders/midfielders; passing accuracy "
                    "and dribbling rank higher.")
        with st.expander("Per-position correlation plot"):
            st.image(str(OUT / "eda_position_correlations.png"))

    with t3:
        bias = pd.read_csv(OUT / "improve_group_bias.csv")
        piv = bias.pivot(index="model", columns="group", values="mean_signed_err_log")
        order = pd.read_csv(OUT / "improve_results.csv").model
        piv = piv.reindex([m for m in order if m in piv.index])
        st.markdown("**Mean signed error (log, prediction − actual)** by group. 0 = unbiased; −0.5 ≈ predicts "
                    "about 40% too low.")
        st.dataframe(piv.style.format("{:+.2f}").background_gradient(cmap="RdBu", vmin=-0.8, vmax=0.8, axis=None))
        st.markdown("Team context nearly removed the Burnley bias (+0.73 → +0.15). Stars (€60M+) are still "
                    "under-predicted.")
        st.markdown("**Ten largest misses from the baseline model**")
        st.dataframe(pd.read_csv(OUT / "model_misses.csv").head(10), hide_index=True)

    with t4:
        st.markdown(
            "- **One season, 383 players.** No time-based split is possible; error estimates are noisy "
            "(especially for the 26 goalkeepers).\n"
            "- **Values are a snapshot.** If collected after the summer window they include information that "
            "postdates the stats.\n"
            "- **Stats can't see everything:** contract length, injuries, club prestige, nationality and hype.\n"
            "- **Tails are compressed:** the model under-predicts the very top and over-predicts the very bottom.\n"
            "- **Market values are Transfermarkt's** and used here for personal, educational purposes only."
        )


PAGES = {
    "Overview": page_overview,
    "Predict a value": page_predict,
    "Model vs market": page_explorer,
    "Scouting table": page_scouting,
    "Insights": page_insights,
}

with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0'>⚽ PL Market Value</h3>", unsafe_allow_html=True)
    st.caption("2025/26 · Transfermarkt vs. stats")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()

PAGES[page]()
