# %% [markdown]
# # Repeated Measures ANOVA
# One-way rm ANOVA comparing **Baseline**, **Physics**, and **GeoCtrl** conditions.
#
# - **Within-subject factor:** condition
# - **Post-hoc:** pairwise t-tests with Bonferroni correction (via pingouin)
#
# > Data are aggregated to one mean per participant × condition before the ANOVA,
# > as rm ANOVA requires a single value per cell.

# %%
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from statannotations.Annotator import Annotator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*reconfigured without applying.*")

_IN_JUPYTER = "ipykernel" in sys.modules

# ── load data if not already in scope or preprocessing is missing ──────────────
_PREPROCESSED_COLS = ["Sum Wrist Rotation", "Sum Wrist Translation",
                      "right_wrist_rotation", "left_wrist_rotation"]
try:
    df_summary
    DATA_ROOT
    assert all(c in df_summary.columns for c in _PREPROCESSED_COLS), "stale df_summary"
except (NameError, AssertionError):
    import runpy, pathlib
    _ns = runpy.run_path(str(pathlib.Path(__file__).parent / "load_data.py"))
    df_summary = _ns["df_summary"]
    DATA_ROOT  = _ns["DATA_ROOT"]


# %% [markdown]
# ## Configuration
# **Change `DV` to any numeric column in `df_summary` to re-run the full analysis.**

# ── dependent variable ────────────────────────────────────────────────────────
# DV = "Task Completion Time"
DV = "Total Die Rotation"
# DV = "Total Die Translation"
# DV = "Sum Wrist Rotation"
# DV = "Sum Wrist Translation"
# DV = "Total Head Rotation"

# Optional: y-axis label override (set to None to use the column name as-is)
DV_LABEL = None

# ── analysis settings ─────────────────────────────────────────────────────────
EXCLUDE_TIMEOUTS = True
CONDITION_ORDER  = ["Baseline", "Physics", "GeoCtrl"]
PALETTE          = {"Baseline": "#4C72B0", "Physics": "#DD8452", "GeoCtrl": "#55A868"}

# Derived
_ylabel    = DV_LABEL if DV_LABEL else DV
_safe_dv   = DV.replace(" ", "_").replace("/", "_")
_plot_file = DATA_ROOT / f"rmanova_{_safe_dv}.png"

print(f"DV: '{DV}'")
if DV not in df_summary.columns:
    raise ValueError(f"Column '{DV}' not found. Available columns:\n{list(df_summary.columns)}")


# %% [markdown]
# ## 1 · Prepare data

df = df_summary[["participant_id", "condition", "Is Timeout", DV]].copy()
df[DV]           = pd.to_numeric(df[DV], errors="coerce")
df["Is Timeout"] = df["Is Timeout"].astype(str).str.lower() == "true"
df["condition"]  = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)

n_total = len(df)
df_clean = df[~df["Is Timeout"]].copy() if EXCLUDE_TIMEOUTS else df.copy()
print(f"Trials: {n_total} total, {n_total - len(df_clean)} excluded → {len(df_clean)} remaining")

# Aggregate to one mean per participant × condition (required for rm ANOVA)
df_agg = (
    df_clean.groupby(["participant_id", "condition"], observed=True)[DV]
    .mean()
    .reset_index()
)
print(f"\nAggregated: {len(df_agg)} cells ({df_agg['participant_id'].nunique()} participants × {df_agg['condition'].nunique()} conditions)")
df_agg.pivot(index="participant_id", columns="condition", values=DV).round(3)


# %% [markdown]
# ## 2 · Descriptive statistics

desc = (
    df_agg.groupby("condition", observed=True)[DV]
    .agg(n="count", mean="mean", sd="std", median="median",
         q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
    .round(3)
)
print(desc)


# %% [markdown]
# ## 3 · Repeated Measures ANOVA


aov = pg.rm_anova(
    data=df_agg,
    dv=DV,
    within="condition",
    subject="participant_id",
    detailed=True,
)
print(aov.to_string())


# %% [markdown]
# ## 4 · Post-hoc pairwise tests (Bonferroni)


posthoc = pg.pairwise_tests(
    data=df_agg,
    dv=DV,
    within="condition",
    subject="participant_id",
    padjust="bonf",
)
posthoc["sig"] = posthoc["p_corr"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
)
print(posthoc[["A", "B", "T", "dof", "p_unc", "p_corr", "sig"]].to_string(index=False))


# %% [markdown]
# ## 5 · Box plot with significance annotations

fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    data=df_clean,
    x="condition", y=DV,
    order=CONDITION_ORDER, palette=PALETTE, width=0.5,
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
    ax=ax,
)
sns.stripplot(
    data=df_clean,
    x="condition", y=DV,
    order=CONDITION_ORDER,
    color="black", size=3, alpha=0.35, jitter=True,
    ax=ax,
)

sig_pairs  = list(zip(posthoc["A"], posthoc["B"]))
sig_labels = list(posthoc["sig"])

annotator = Annotator(ax, sig_pairs, data=df_clean,
                      x="condition", y=DV, order=CONDITION_ORDER)
annotator.configure(line_width=1.2, text_format="simple", fontsize=11)
annotator.set_custom_annotations(sig_labels)
annotator.annotate()

ax.set_title(f"{DV} by Condition", fontsize=13, pad=12)
ax.set_xlabel("Condition", fontsize=11)
ax.set_ylabel(_ylabel, fontsize=11)
ax.tick_params(axis="x", labelsize=11)

# rm ANOVA result footnote
row = aov[aov["Source"] == "condition"].iloc[0]
note = (f"rm ANOVA: F({int(row['ddof1'])},{int(row['ddof2'])}) = {row['F']:.3f}, "
        f"p = {row['p-unc']:.4f}, η²p = {row['np2']:.3f}")
ax.text(0.98, 0.02, note,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(_plot_file, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_plot_file.name}")

# %%
