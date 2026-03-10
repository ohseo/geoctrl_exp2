# %% [markdown]
# # LMM Analysis
# Linear Mixed Model comparing **Baseline**, **Physics**, and **GeoCtrl** conditions.
#
# - **Fixed effect:** condition (treatment-coded, Baseline as reference)
# - **Random effect:** by-participant intercept
# - **Post-hoc:** pairwise comparisons with Bonferroni correction

# %%
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statannotations.Annotator import Annotator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*reconfigured without applying.*")
warnings.filterwarnings("ignore", message=".*Random effects covariance is singular.*")

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
    try:
        _here = pathlib.Path(__file__).parent
    except NameError:
        _here = pathlib.Path().resolve()
    _ns = runpy.run_path(str(_here / "load_data.py"))
    df_summary = _ns["df_summary"]
    DATA_ROOT  = _ns["DATA_ROOT"]

# ── merge expertise columns from df_participants ───────────────────────────────
_EXPERTISE_COLS = ["headset_usage", "bare_hand_expertise", "direct_manipulation_expertise"]
if not all(c in df_summary.columns for c in _EXPERTISE_COLS):
    import runpy, pathlib
    try:
        _here = pathlib.Path(__file__).parent
    except NameError:
        _here = pathlib.Path().resolve()
    _ep_ns = runpy.run_path(str(_here / "extract_participants.py"))
    _df_p  = _ep_ns["df_participants"][["participant_id"] + _EXPERTISE_COLS].copy()
    for c in _EXPERTISE_COLS:
        _df_p[c] = pd.to_numeric(_df_p[c], errors="coerce")
    df_summary = df_summary.merge(_df_p, on="participant_id", how="left")


# %% [markdown]
# ## Configuration
# **Change `DV` to any numeric column in `df_summary` to re-run the full analysis.**

# ── dependent variable ────────────────────────────────────────────────────────
DV = "Task Completion Time"
# DV = "Total Die Rotation"
# DV = "Total Die Translation"
# DV = "Sum Wrist Rotation"
# DV = "Sum Wrist Translation"
# DV = "Total Head Rotation"
# DV = "Total Head Translation"
# DV = "Target Offset Rotation Angle"
# DV = "Target Offset Position Magnitude"

# Optional: y-axis label override (set to None to use the column name as-is)
DV_LABEL = None

# ── analysis settings ─────────────────────────────────────────────────────────
EXCLUDE_TIMEOUTS = False   # drop timed-out trials (only meaningful for TCT)
EXCLUDE_OUTLIERS = False
EXCLUDE_NON_EXPERTS = False
POSTHOC  = "ttest_rel"   # "ttest_rel" = paired t-test on participant means (within-subjects)
                         # "ttest_ind" = independent samples t-test on raw trials
IV1       = "condition"  # primary fixed effect column
IV1_ORDER = ["Baseline", "GeoCtrl", "Physics"]  # level order; also sets reference (first item)
IV2       = None         # second IV for interaction, e.g. "Block Num" or "Trial Num"
                         # set to None for main-effect-only model
ORDER   = ["Baseline", "GeoCtrl", "Physics"]   # condition order for PALETTE / plots
PALETTE = {"Baseline": "#4C72B0", "Physics": "#DD8452", "GeoCtrl": "#55A868"}
PILOTS = ['P0','P99','P26','P30']
VALUE_OUTLIERS = ['P23','P18','P4']  # participant IDs to exclude, e.g. ["P99"]
NON_EXPERTS = ['P1','P2','P23','P21','P7','P19','P8']
if EXCLUDE_OUTLIERS:
    OUTLIERS = PILOTS + VALUE_OUTLIERS
else:
    OUTLIERS = PILOTS
if EXCLUDE_NON_EXPERTS:
    OUTLIERS = OUTLIERS + NON_EXPERTS

# Derived
_ylabel    = DV_LABEL if DV_LABEL else DV
_safe_dv   = DV.replace(" ", "_").replace("/", "_")
_plot_file = DATA_ROOT / f"lmm_{_safe_dv}.png"
_ref       = IV1_ORDER[0]
_iv1_term  = IV1 if " " not in IV1 else f"Q('{IV1}')"
_cond_term = f"C({_iv1_term}, Treatment(reference='{_ref}'))"
_palette   = PALETTE if IV1 == "condition" else "Set2"
_formula   = (
    f"Q('{DV}') ~ {_cond_term} * C(Q('{IV2}'))"
    if IV2 else
    f"Q('{DV}') ~ {_cond_term}"
)
print(f"IV1: '{IV1}'  |  Formula: {_formula}")

print(f"DV: '{DV}'")
if DV not in df_summary.columns:
    raise ValueError(f"Column '{DV}' not found. Available columns:\n{list(df_summary.columns)}")


# %% [markdown]
# ## 1 · Prepare data


df = df_summary[["participant_id", "condition", "Block Num", "Trial Num",
                  "Is Timeout", "direct_manipulation_expertise", DV]].copy()

df[DV]           = pd.to_numeric(df[DV], errors="coerce")
df["Is Timeout"] = df["Is Timeout"].astype(str).str.lower() == "true"
df[IV1]          = pd.Categorical(df[IV1], categories=IV1_ORDER, ordered=True)

n_total = len(df)
df_intime  = df[~df["Is Timeout"]].copy() if EXCLUDE_TIMEOUTS else df.copy()
df_lmm = df_intime[~df_intime['participant_id'].isin(OUTLIERS)].copy()
n_excluded = n_total - len(df_lmm)
print(f"Trials: {n_total} total, {n_excluded} excluded → {len(df_lmm)} for LMM")

df_lmm.groupby(["participant_id", IV1])[DV].agg(["count", "mean", "std"]).round(3)
print(f"{df_lmm["participant_id"].unique()}")


# %% [markdown]
# ## 2 · Descriptive statistics

desc = (
    df_lmm.groupby(IV1, observed=True)[DV]
    .agg(n="count", mean="mean", sd="std", median="median",
         q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
    .round(3)
)
print(desc)


# %% [markdown]
# ## 3 · Linear Mixed Model

model = smf.mixedlm(_formula, data=df_lmm, groups=df_lmm["participant_id"])
result = model.fit(method="lbfgs", reml=True)
print(result.summary())


# %% [markdown]
# ## 4 · Post-hoc pairwise comparisons (Bonferroni)

from itertools import combinations as _combinations
pairs = list(_combinations(IV1_ORDER, 2))

posthoc_rows = []
for c1, c2 in pairs:
    g1 = df_lmm.loc[df_lmm[IV1] == c1, DV]
    g2 = df_lmm.loc[df_lmm[IV1] == c2, DV]
    t_stat, p_raw = stats.ttest_ind(g1, g2)
    cohens_d = (g1.mean() - g2.mean()) / np.sqrt(
        ((len(g1) - 1) * g1.std()**2 + (len(g2) - 1) * g2.std()**2)
        / (len(g1) + len(g2) - 2)
    )
    posthoc_rows.append({
        "pair":      f"{c1} vs {c2}",
        "mean_diff": round(g1.mean() - g2.mean(), 3),
        "t":         round(t_stat, 3),
        "p_raw":     round(p_raw, 4),
        "cohens_d":  round(cohens_d, 3),
    })

df_posthoc = pd.DataFrame(posthoc_rows)
df_posthoc["p_bonf"] = (df_posthoc["p_raw"] * len(pairs)).clip(upper=1.0).round(4)
df_posthoc["sig"] = df_posthoc["p_bonf"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
)
print(df_posthoc.to_string(index=False))


# %% [markdown]
# ## 5 · Box plot with significance annotations

fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    data=df_lmm, x=IV1, y=DV,
    order=IV1_ORDER, palette=_palette, width=0.5,
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
    ax=ax,
)
sns.stripplot(
    data=df_lmm, x=IV1, y=DV,
    order=IV1_ORDER,
    color="black", size=3, alpha=0.35, jitter=True,
    ax=ax,
)

sig_pairs  = [(r["pair"].split(" vs ")[0], r["pair"].split(" vs ")[1])
              for _, r in df_posthoc.iterrows()]
sig_labels = list(df_posthoc["sig"])

annotator = Annotator(ax, sig_pairs, data=df_lmm,
                      x=IV1, y=DV, order=IV1_ORDER)
annotator.configure(line_width=1.2, text_format="simple", fontsize=11)
annotator.set_custom_annotations(sig_labels)
annotator.annotate()

ax.set_title(f"{DV} by {IV1}", fontsize=13, pad=12)
ax.set_xlabel(IV1, fontsize=11)
ax.set_ylabel(_ylabel, fontsize=11)
ax.tick_params(axis="x", labelsize=11)

# LMM result footnote
lmm_fe  = result.fe_params
p_vals  = result.pvalues
note_lines = ["LMM fixed effects (ref = Baseline):"]
for k, v in lmm_fe.items():
    if k == "Intercept":
        continue
    label = k.replace(f"{_cond_term}[T.", "").rstrip("]")
#     note_lines.append(f"  {label}: β={v:.3f}, p={p_vals[k]:.4f}")
# ax.text(0.98, 0.02, "\n".join(note_lines),
#         transform=ax.transAxes, fontsize=8,
#         verticalalignment="bottom", horizontalalignment="right",
#         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.tight_layout()

# Config stamp — shown at the bottom of the saved image
_cfg_text = (
    f"exclude_timeouts={EXCLUDE_TIMEOUTS} | exclude_outliers={EXCLUDE_OUTLIERS} |   "
    f"exclude_non_experts={EXCLUDE_NON_EXPERTS} | posthoc={POSTHOC} |   "
    f"outliers={OUTLIERS}"
)
fig.text(0.5, -0.02, _cfg_text, ha="center", va="top",
         fontsize=6.5, color="gray", style="italic",
         wrap=True)

plt.savefig(_plot_file, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_plot_file.name}")


# %% [markdown]
# ## 6 · Random effects summary

# %%
re_summary = pd.DataFrame(
    {pid: vals for pid, vals in result.random_effects.items()}
).T.rename(columns={0: "intercept_re"}).round(3)
print("Per-participant random intercepts:")
print(re_summary)

icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
print(f"\nICC (participant variance / total variance) = {icc:.3f}")


# %% [markdown]
# ## 7 · Grouped box plot
# Set `GRP_X` and `GRP_HUE` to any columns in `df_lmm`.
# Set the corresponding `_ORDER` list to `None` to auto-sort.

GRP_X        = "condition"  # x-axis grouping
GRP_X_ORDER  = ORDER   # list of levels in display order, or None to auto-sort numerically

GRP_HUE      = "direct_manipulation_expertise"   # hue (colour) grouping
GRP_HUE_ORDER = None        # list of levels, or None to auto-sort

# ── derive orders if not specified ────────────────────────────────────────────
_df_grp = df_lmm[[GRP_X, GRP_HUE, DV]].dropna(subset=[GRP_X, GRP_HUE, DV]).copy()

# Cast numeric-looking columns to int then str so they sort correctly
for _col in [GRP_X, GRP_HUE]:
    try:
        _df_grp[_col] = _df_grp[_col].astype(float).astype(int).astype(str)
    except (ValueError, TypeError):
        _df_grp[_col] = _df_grp[_col].astype(str)

def _auto_order(col, override):
    if override is not None:
        return [str(v) for v in override]
    vals = _df_grp[col].unique()
    try:
        return sorted(vals, key=lambda v: float(v))
    except ValueError:
        return sorted(vals)

_x_order   = _auto_order(GRP_X,   GRP_X_ORDER)
_hue_order = _auto_order(GRP_HUE, GRP_HUE_ORDER)
_hue_palette = PALETTE if GRP_HUE == "condition" else "Set2"

# ── plot ──────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(max(6, len(_x_order) * 2.2), 5))

sns.boxplot(
    data=_df_grp, x=GRP_X, y=DV,
    hue=GRP_HUE, hue_order=_hue_order, order=_x_order,
    palette=_hue_palette, width=0.6,
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    ax=ax2,
)
sns.stripplot(
    data=_df_grp, x=GRP_X, y=DV,
    hue=GRP_HUE, hue_order=_hue_order, order=_x_order,
    palette=_hue_palette, size=3, alpha=0.4, jitter=True,
    dodge=True, legend=False,
    ax=ax2,
)

ax2.set_title(f"{DV}  |  {GRP_X} × {GRP_HUE}", fontsize=13, pad=10)
ax2.set_xlabel(GRP_X, fontsize=11)
ax2.set_ylabel(_ylabel, fontsize=11)
ax2.legend(title=GRP_HUE, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax2.tick_params(axis="x", labelsize=10)

_cfg_text2 = (
    f"DV={DV}  |  x={GRP_X}  |  hue={GRP_HUE}  |  "
    f"exclude_timeouts={EXCLUDE_TIMEOUTS}  |  outliers={OUTLIERS}"
)
fig2.text(0.5, -0.02, _cfg_text2, ha="center", va="top",
          fontsize=6.5, color="gray", style="italic")

plt.tight_layout()
_safe_grp = f"{GRP_X.replace(' ','_')}_x_{GRP_HUE.replace(' ','_')}"
_grp_plot_file = DATA_ROOT / f"lmm_{_safe_dv}_{_safe_grp}.png"
plt.savefig(_grp_plot_file, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_grp_plot_file.name}")

# %%
