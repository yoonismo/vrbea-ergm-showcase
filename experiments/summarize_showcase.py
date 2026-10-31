"""summarize_showcase.py -- MPLE / MCMC-MLE summary (bias, mean, median, SE, 0.05, 0.95, time)

usage:
    python -m experiments.summarize_showcase              # newest result CSV in log_dir
    python -m experiments.summarize_showcase path/to.csv  # a specific file
"""
import os
import sys
import glob
import pandas as pd

from ergm_showcase import summary_stat
from experiments import config as cfg

# column names in the result CSV: (theta column, time column)
ESTIMATORS = {"MPLE":     ("theta_mple", "time_mple"),
              "MCMC-MLE": ("theta_mle",  "time_mle")}


def summarize(csv_path, parameter, active_terms):
    df = pd.read_csv(csv_path)
    tables = {}
    for name, (theta_col, time_col) in ESTIMATORS.items():
        sub = df[[theta_col, time_col]].rename(columns={time_col: "time"})
        tables[name] = summary_stat(sub, parameter, active_terms)
    return pd.concat(tables, axis=1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        results = [p for p in glob.glob(os.path.join(cfg.log_dir, "*.csv"))
                   if not p.endswith("_summary.csv")]
        csv_path = max(results, key=os.path.getmtime)

    table = summarize(csv_path, cfg.param, cfg.active)
    print(f"\n{os.path.basename(csv_path)}\n")
    print(table.to_string())

    out_path = csv_path.replace(".csv", "_summary.csv")
    table.to_csv(out_path)
    print(f"\nsaved: {out_path}")
