# %% [markdown]
# # geoctrl_exp2 — Data Loading
# Loads **EventLog** and **Summary** CSVs for statistical analysis.
# StreamData is included as an optional loader at the bottom.
#
# **Conditions:** `Baseline_X`, `Physics_Y`, `GeoCtrl_Z`
# **File types:** Summary (per condition), EventLog (per condition), StreamData (per trial)

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root whether running as a script or in Jupyter
try:
    DATA_ROOT = Path(__file__).parent
except NameError:
    DATA_ROOT = Path().resolve()  # cwd when running interactively

print("Data root:", DATA_ROOT)

# %% [markdown]
# ## Helpers

def _parse_file_meta(filepath: Path) -> dict:
    """Extract participant_id and condition from a CSV file path."""
    # Parent folder: P0_오서영  →  P0
    p_match = re.match(r"(P\d+)", filepath.parent.name)
    participant_id = p_match.group(1) if p_match else filepath.parent.name

    name = filepath.stem
    if "Baseline" in name:
        condition = "Baseline"
    elif "Physics" in name:
        condition = "Physics"
    elif "GeoCtrl" in name:
        condition = "GeoCtrl"
    else:
        condition = "Unknown"

    return {"participant_id": participant_id, "condition": condition}


def _load_csvs(pattern: str, extra_meta_fn=None) -> pd.DataFrame:
    """
    Glob *pattern* under DATA_ROOT, read each CSV, prepend metadata columns,
    and concatenate by column name (handles different column orders across conditions).

    Parameters
    ----------
    pattern : str
        rglob pattern, e.g. "*_Summary.csv"
    extra_meta_fn : callable, optional
        Called with the Path; should return a dict of additional metadata columns.
    """
    files = sorted(DATA_ROOT.rglob(pattern))
    if not files:
        print(f"[warn] No files found for pattern: {pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        meta = _parse_file_meta(f)
        if extra_meta_fn:
            meta.update(extra_meta_fn(f))

        df = pd.read_csv(f, encoding="utf-8-sig")

        # Prepend metadata as leftmost columns
        for col, val in reversed(meta.items()):
            df.insert(0, col, val)

        dfs.append(df)

    # concat aligns on column names, fills missing with NaN
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined


# %% [markdown]
# ## Summary data
# One row per trial. Primary metric source.

df_summary = _load_csvs("*_Summary.csv")

df_summary["Block Num"] = df_summary["Block Num"].astype(int).astype(str)
df_summary["Trial Num"] = df_summary["Trial Num"].astype(int).astype(str)

print(f"Summary: {df_summary.shape[0]} rows × {df_summary.shape[1]} cols")
print(df_summary.groupby(["participant_id", "condition"]).size().rename("n_trials"))

# %%
df_summary.head()


# %% [markdown]
# ## Preprocessing — unified wrist columns
# Baseline & GeoCtrl log as `Right/Left Total Wrist X`.
# Physics logs as `Total Wrist X` (right) / `Total Wrist X_left`.
# The cells below coalesce both naming schemes into four consistent columns.


df_summary["right_wrist_rotation"] = (
    df_summary["Right Total Wrist Rotation"]
    .fillna(df_summary["Total Wrist Rotation"])
)
df_summary["left_wrist_rotation"] = (
    df_summary["Left Total Wrist Rotation"]
    .fillna(df_summary["Total Wrist Rotation_left"])
)
df_summary["right_wrist_translation"] = (
    df_summary["Right Total Wrist Translation"]
    .fillna(df_summary["Total Wrist Translation"])
)
df_summary["left_wrist_translation"] = (
    df_summary["Left Total Wrist Translation"]
    .fillna(df_summary["Total Wrist Translation_left"])
)

df_summary["Sum Wrist Rotation"]    = df_summary["right_wrist_rotation"]    + df_summary["left_wrist_rotation"]
df_summary["Sum Wrist Translation"] = df_summary["right_wrist_translation"] + df_summary["left_wrist_translation"]

# Normalise target offset rotation angle to [0, 180]: values > 180 → 360 - value
df_summary["Target Offset Rotation Angle"] = pd.to_numeric(
    df_summary["Target Offset Rotation Angle"], errors="coerce"
)
df_summary["Target Offset Rotation Angle"] = df_summary["Target Offset Rotation Angle"].where(
    df_summary["Target Offset Rotation Angle"] <= 180,
    360 - df_summary["Target Offset Rotation Angle"],
)

# Verify: no NaNs expected after coalescing
_wrist_cols = ["right_wrist_rotation", "left_wrist_rotation",
               "right_wrist_translation", "left_wrist_translation",
               "Sum Wrist Rotation", "Sum Wrist Translation"]
print("NaN counts after coalescing:")
print(df_summary[_wrist_cols].isna().sum())
print("\nMeans by condition:")
print(df_summary.groupby("condition")[_wrist_cols].mean().round(2))


# %% [markdown]
# ## Condition-specific subsets

df_baseline = df_summary[df_summary["condition"] == "Baseline"].reset_index(drop=True)
df_physics  = df_summary[df_summary["condition"] == "Physics"].reset_index(drop=True)
df_geoctrl  = df_summary[df_summary["condition"] == "GeoCtrl"].reset_index(drop=True)

print(f"df_baseline : {len(df_baseline)} rows")
print(f"df_physics  : {len(df_physics)} rows")
print(f"df_geoctrl  : {len(df_geoctrl)} rows")

# %%
df_geoctrl.head()


# %% [markdown]
# ## EventLog data
# One row per sub-trial event (Trial Load / Start / Grab / Release / On Target / Off Target / End).

df_events = _load_csvs("*_EventLog.csv")

print(f"EventLog: {df_events.shape[0]} rows × {df_events.shape[1]} cols")
print(df_events.groupby(["participant_id", "condition", "Event Name"]).size().rename("n_events"))

# %%
df_events.head()


# %% [markdown]
# ## NASA-TLX data
# One row per participant × condition. Data starts at row 6 (5 header/metadata lines).
# Task column: X = Baseline, Y = Physics, Z = GeoCtrl.

_TASK_TO_CONDITION = {"X": "Baseline", "Y": "Physics", "Z": "GeoCtrl"}
_TLX_SUBSCALES = ["Mental Demand", "Physical Demand", "Temporal Demand",
                  "Performance", "Effort", "Frustration"]

_tlx_files = sorted((DATA_ROOT / "TLX").glob("*.csv"))
_tlx_dfs = []
for f in _tlx_files:
    pid_match = re.search(r"_(P\d+)\.csv$", f.name)
    if not pid_match:
        continue
    pid = pid_match.group(1)
    _df = pd.read_csv(f, skiprows=5, encoding="utf-8-sig")
    _df.insert(0, "participant_id", pid)
    _df["condition"] = _df["Task"].map(_TASK_TO_CONDITION)
    _tlx_dfs.append(_df)

df_tlx = pd.concat(_tlx_dfs, ignore_index=True) if _tlx_dfs else pd.DataFrame()

# Coerce subscales to numeric and compute unweighted TLX total (mean of 6 subscales)
for col in _TLX_SUBSCALES:
    df_tlx[col] = pd.to_numeric(df_tlx[col], errors="coerce")
df_tlx["TLX Total"] = df_tlx[_TLX_SUBSCALES].mean(axis=1).round(2)

df_tlx = df_tlx[["participant_id", "condition"] + _TLX_SUBSCALES + ["TLX Total"]]

print(f"TLX: {len(df_tlx)} rows ({df_tlx['participant_id'].nunique()} participants × 3 conditions)")
print(df_tlx.groupby("condition")[["TLX Total"] + _TLX_SUBSCALES].mean().round(2))

# %%
df_tlx


# %% [markdown]
# ## Optional: StreamData
# High-frequency per-frame data (one file per trial).
# Uncomment the cell below only when needed — loading 180 files takes a moment.

def _stream_meta(filepath: Path) -> dict:
    """Extract block and trial numbers from a StreamData filename."""
    m = re.search(r"Block(\d+)_Trial(\d+)", filepath.stem)
    return {
        "block_num": int(m.group(1)) if m else np.nan,
        "trial_num": int(m.group(2)) if m else np.nan,
    }

# Uncomment to load:
# df_stream = _load_csvs("*_StreamData.csv", extra_meta_fn=_stream_meta)
# print(f"StreamData: {df_stream.shape[0]} rows × {df_stream.shape[1]} cols")
# df_stream.head()


# %% [markdown]
# ## Quick sanity checks

# Column names in each combined dataframe
print("=== df_summary columns ===")
print(list(df_summary.columns))
print("\n=== df_events columns ===")
print(list(df_events.columns))

# Summary: trial counts per participant × condition
df_summary.groupby(["participant_id", "condition"])["Trial Num"].count().unstack()

# Summary: mean task completion time
df_summary.groupby(["participant_id", "condition"])["Task Completion Time"].mean().unstack().round(3)

# Summary: timeout rate
df_summary.groupby(["condition"])["Is Timeout"].apply(lambda s: s.astype(str).str.lower().eq("true").mean()).rename("timeout_rate").round(4)

# EventLog: unique event types per condition
df_events.groupby("condition")["Event Name"].unique()


# %% [markdown]
# ## Participant-level outlier check
# Computes per-participant summary statistics and flags outliers using
# the IQR method (value < Q1 − 1.5·IQR  or  value > Q3 + 1.5·IQR).
# Change `OUTLIER_VARS` to inspect different variables.

OUTLIER_VARS = {
    "Task Completion Time": "mean",   # average TCT across all trials
    "Is Timeout":           "sum",    # total number of timed-out trials
    "Sum Wrist Rotation":   "mean",
    "Total Die Rotation":   "mean",
}

_tco = df_summary.copy()
_tco["Is Timeout"] = _tco["Is Timeout"].astype(str).str.lower() == "true"
for col in OUTLIER_VARS:
    _tco[col] = pd.to_numeric(_tco[col], errors="coerce")

# Aggregate per participant (collapse across conditions and trials)
_agg_funcs = {col: fn for col, fn in OUTLIER_VARS.items()}
df_participant = _tco.groupby("participant_id").agg(_agg_funcs).round(3)

# IQR-based outlier flag per variable
def _iqr_flag(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

df_outlier_flags = df_participant.apply(_iqr_flag)
df_outlier_flags.columns = [f"{c}_outlier" for c in df_outlier_flags.columns]

df_participant_report = pd.concat([df_participant, df_outlier_flags], axis=1)

print("=== Per-participant stats ===")
print(df_participant_report.to_string())

print("\n=== Outlier summary ===")
flag_cols = [c for c in df_participant_report.columns if c.endswith("_outlier")]
flagged = df_participant_report[flag_cols]
if flagged.any().any():
    for var in flag_cols:
        who = flagged.index[flagged[var]].tolist()
        if who:
            src = var.replace("_outlier", "")
            print(f"  {src}: {who}")
else:
    print("  No outliers detected.")

# %%
