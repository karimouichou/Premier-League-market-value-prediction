"""
Steps 1-3: audit, clean, and engineer features for the PL market-value project.

Usage:
    python 01_clean_and_features.py PL_players_stats_ready.csv

Outputs (next to the input file's working directory):
    pl_features.csv    modelling table: identifiers + target + engineered features
    pl_audit.txt       short audit report (what was dropped/filled and why)
"""
import html
import sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "PL_players_stats_ready.csv"
MIN_MINUTES = 450  # per-90 stats below this are mostly noise

report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


# ----------------------------------------------------------------------------
# STEP 1 - LOAD AND AUDIT
# ----------------------------------------------------------------------------
df = pd.read_csv(SRC, encoding="iso-8859-1")


def fix_mojibake(s):
    """The file holds UTF-8 text that was decoded as latin-1 ('AndrÃ©s')."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s


for col in ("player_name", "team_name"):
    # also decode HTML entities left over from scraping ("Brighton &amp; Hove Albion")
    df[col] = df[col].map(fix_mojibake).map(lambda s: html.unescape(s) if isinstance(s, str) else s)

log("=== STEP 1: AUDIT ===")
log(f"Rows x cols: {df.shape}")
log(f"Duplicate rows: {df.duplicated().sum()} | duplicate player_id: {df.player_id.duplicated().sum()}")
log(f"Teams: {df.team_name.nunique()}")
log("Players per position (G/D/M/F):")
log(df.position_stats.value_counts().to_string())
log("Market value (EUR) summary:")
log(df.market_value_eur.describe().round(0).to_string())
log(f"Zero/negative values: {(df.market_value_eur <= 0).sum()} | "
    f"skew raw={df.market_value_eur.skew():.2f}, log={np.log1p(df.market_value_eur).skew():.2f}")
log(f"Players under {MIN_MINUTES} min: {(df.minutesPlayed < MIN_MINUTES).sum()}")
na = df.isna().sum()
log("Columns with missing values:")
log(na[na > 0].to_string())

# ----------------------------------------------------------------------------
# STEP 2 - CLEAN AND FILTER
# ----------------------------------------------------------------------------
log("\n=== STEP 2: CLEAN & FILTER ===")

# Missing xG/xA means the player had no shots / chances tracked -> true zero.
# goalsPrevented only exists for goalkeepers -> 0 for everyone else.
for col in ("expectedGoals", "expectedAssists", "goalsPrevented"):
    df[col] = df[col].fillna(0.0)
log("Filled NaN in expectedGoals / expectedAssists / goalsPrevented with 0 "
    "(no shots tracked, or goalkeeper-only metric)")

n_before = len(df)
df = df[df.minutesPlayed >= MIN_MINUTES].copy()
log(f"Kept players with >= {MIN_MINUTES} minutes: {len(df)} of {n_before}")
log("Remaining per position:")
log(df.position_stats.value_counts().to_string())
if (df.position_stats == "G").sum() < 30:
    log("WARNING: fewer than 30 goalkeepers -> too few for a separate GK model; "
        "keep GK in the shared model with position interactions.")

# Redundant / non-informative columns
DROP_REDUNDANT = [
    "id",                    # internal id (player_id kept as reference only)
    "goalsAssistsSum",       # goals + assists
    "inaccuratePasses",      # totalPasses - accuratePasses
    "scoringFrequency",      # minutes per goal, derived from goals/minutes
    "attemptPenaltyMiss",    # covered by penaltiesTaken / penaltyGoals
    "attemptPenaltyTarget",
    "leftFootGoals",         # footedness of goals is not a value driver
    "rightFootGoals",
    "tacklesWon",            # overlaps with tackles
    "totalContest",
    "totwAppearances",       # team-of-the-week count leaks the rating signal
]
df = df.drop(columns=[c for c in DROP_REDUNDANT if c in df.columns])
log(f"Dropped redundant columns: {DROP_REDUNDANT}")

# ----------------------------------------------------------------------------
# STEP 3 - FEATURE ENGINEERING
# ----------------------------------------------------------------------------
log("\n=== STEP 3: FEATURE ENGINEERING ===")

ID_COLS = ["player_id", "player_name", "team_name", "position_stats"]
TARGET_RAW = "market_value_eur"
KEEP_AS_IS = ["age", "appearances", "matchesStarted", "minutesPlayed", "rating"]

# Rate / percentage columns stay as they are (already normalised)
RATE_COLS = [c for c in df.columns
             if c.endswith("Percentage") or c in ("goalConversionPercentage",
                                                  "penaltyConversion", "setPieceConversion")]

# Everything else numeric is a counting stat -> per 90 minutes
EXCLUDE = set(ID_COLS + [TARGET_RAW] + KEEP_AS_IS + RATE_COLS + ["cleanSheet"])
COUNT_COLS = [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]

# Build columns in a dict and assemble once (avoids fragmented-frame warnings)
feat = {c: df[c] for c in ID_COLS}
feat["position"] = df["position_stats"].map({"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"})

# Context
feat["age"] = df["age"]
feat["age_sq"] = df["age"] ** 2                      # captures the age peak
feat["minutes"] = df["minutesPlayed"]
feat["appearances"] = df["appearances"]
feat["start_ratio"] = df["matchesStarted"] / df["appearances"].clip(lower=1)
feat["rating"] = df["rating"]

# Per-90 counting stats
for c in COUNT_COLS:
    feat[f"{c}_p90"] = df[c] / df["minutesPlayed"] * 90

# Clean sheets are a team result -> rate per appearance, not per 90
feat["cleanSheet_rate"] = df["cleanSheet"] / df["appearances"].clip(lower=1)

# Percentage columns
for c in RATE_COLS:
    feat[c] = df[c]

# Composite, position-relevant features
feat["goal_contrib_p90"] = (df["goals"] + df["assists"]) / df["minutesPlayed"] * 90
feat["xG_xA_p90"] = (df["expectedGoals"] + df["expectedAssists"]) / df["minutesPlayed"] * 90
feat["def_actions_p90"] = (df["tackles"] + df["interceptions"] + df["clearances"]
                           + df["ballRecovery"]) / df["minutesPlayed"] * 90
feat["chance_creation_p90"] = (df["keyPasses"] + df["bigChancesCreated"]) / df["minutesPlayed"] * 90
feat["xG_overperf_p90"] = (df["goals"] - df["expectedGoals"]) / df["minutesPlayed"] * 90

# One-hot position
for p in ("GK", "DEF", "MID", "FWD"):
    feat[f"pos_{p}"] = (feat["position"] == p).astype(int)

# Position x stat interactions: encode "this stat matters for this position".
# Each stat is only active (non-zero) for the positions listed.
KEY_STATS = {
    "FWD": ["goals_p90", "expectedGoals_p90", "shotsOnTarget_p90", "goal_contrib_p90", "bigChancesMissed_p90"],
    "MID": ["ballRecovery_p90", "keyPasses_p90", "expectedAssists_p90", "chance_creation_p90",
            "accurateFinalThirdPasses_p90"],
    "DEF": ["cleanSheet_rate", "ballRecovery_p90", "interceptions_p90", "clearances_p90",
            "def_actions_p90"],
    "GK":  ["cleanSheet_rate", "saves_p90", "goalsPrevented_p90"],
}
for pos, cols in KEY_STATS.items():
    for c in cols:
        feat[f"{pos}_x_{c}"] = feat[f"pos_{pos}"] * feat[c]

# Target
feat["market_value_eur"] = df[TARGET_RAW]
feat["log_value"] = np.log1p(df[TARGET_RAW])
feat = pd.DataFrame(feat)

# Sanity checks
assert feat.drop(columns=["position", "player_name", "team_name", "position_stats"]) \
           .replace([np.inf, -np.inf], np.nan).isna().sum().sum() == 0, "NaN/inf in features"
assert feat.player_id.is_unique

n_features = feat.shape[1] - len(ID_COLS) - 3  # minus position, market_value_eur, log_value
log(f"Per-90 counting stats: {len(COUNT_COLS)} | percentage columns: {len(RATE_COLS)}")
log(f"Interaction features: {sum(len(v) for v in KEY_STATS.values())}")
log(f"Final table: {feat.shape[0]} players x {feat.shape[1]} columns (~{n_features} features)")

# Quick check that the position-specific idea shows up in the data
log("\nSpearman correlation with log_value, by position (top 5 each):")
num = feat.select_dtypes("number").drop(columns=["player_id", "market_value_eur", "log_value"])
num = num.loc[:, ~num.columns.str.contains("_x_|^pos_")]
for pos in ("GK", "DEF", "MID", "FWD"):
    sub = feat[feat.position == pos]
    block = num.loc[sub.index]
    block = block.loc[:, block.nunique() > 1]  # skip columns constant within the position
    corr = block.corrwith(sub["log_value"], method="spearman").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    log(f"  {pos}: " + ", ".join(f"{k} ({v:+.2f})" for k, v in corr.head(5).items()))

feat.to_csv("pl_features.csv", index=False, encoding="utf-8")
with open("pl_audit.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))
log("\nSaved pl_features.csv and pl_audit.txt")
