"""Run the whole pipeline: cleaning/features -> EDA -> modelling.

    python run_all.py

Reads  data/PL_players_stats_ready.csv
Writes everything to outputs/ (features table, audit, plots, result tables).
Takes about 10 minutes (steps 3 and 4 run repeated cross-validation).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC, DATA, OUT = ROOT / "src", ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

steps = [
    ("1/4 Clean data + build features", "01_clean_and_features.py", DATA / "PL_players_stats_ready.csv"),
    ("2/4 Exploratory plots",           "02_eda.py",                OUT / "pl_features.csv"),
    ("3/4 Models + evaluation",         "03_model.py",              OUT / "pl_features.csv"),
    ("4/4 Improvements (team context, tuning; XGBoost/SHAP if installed)",
                                        "04_improve.py",            OUT / "pl_features.csv"),
]
for title, script, arg in steps:
    print(f"\n===== {title} =====", flush=True)
    r = subprocess.run([sys.executable, str(SRC / script), str(arg)], cwd=OUT)
    if r.returncode != 0:
        sys.exit(f"Step failed: {script}")
print(f"\nDone. Results are in: {OUT}")
