# %% [markdown]
# # NASA-TLX Analysis
# LMM + box plots with significance annotations for TLX Total and each subscale.
# - **LMM:** condition as fixed effect, participant as random intercept
# - **Post-hoc:** paired t-tests with Bonferroni correction

# %%
import sys
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statannotations.Annotator import Annotator
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*reconfigured without applying.*")
warnings.filterwarnings("ignore", message=".*Random effects covariance is singular.*")

_IN_JUPYTER = "ipykernel" in sys.modules

# ── load data if not already in scope ─────────────────────────────────────────
try:
    df_tlx
    DATA_ROOT
except NameError:
    import runpy, pathlib
    _ns = runpy.run_path(str(pathlib.Path(__file__).parent / "load_data.py"))
    df_tlx    = _ns["df_tlx"]
    DATA_ROOT = _ns["DATA_ROOT"]


# %% [markdown]
# ## Configuration
EXCLUDE_NON_EXPERTS = False
EXCLUDE_OUTLIERS = False

ORDER   = ["Baseline", "GeoCtrl", "Physics"]
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

SUBSCALES = ["Mental Demand", "Physical Demand", "Temporal Demand",
             "Performance", "Effort", "Frustration"]
ALL_VARS  = ["TLX Total"] + SUBSCALES   # Total first, then subscales

PAIRS = list(combinations(ORDER, 2))    # all 3 pairwise combinations


# %% [markdown]
# ## 1 · Prepare data

df = df_tlx[~df_tlx["participant_id"].isin(OUTLIERS)].copy()
df["condition"] = pd.Categorical(df["condition"], categories=ORDER, ordered=True)

print(f"Participants: {sorted(df['participant_id'].unique())}")
print(f"N = {df['participant_id'].nunique()}")
print()
print(df.groupby("condition")[ALL_VARS].mean().round(2))


# %% [markdown]
# ## 2 · Linear Mixed Model (per variable)

_REF = ORDER[0]   # first condition in ORDER is the LMM reference level

def _run_lmm(df: pd.DataFrame, dv: str):
    """Fit LMM with condition as fixed effect and participant as random intercept."""
    model = smf.mixedlm(
        f"Q('{dv}') ~ C(condition, Treatment(reference='{_REF}'))",
        data=df,
        groups=df["participant_id"],
    )
    return model.fit(method="lbfgs", reml=True)

lmm_results = {var: _run_lmm(df, var) for var in ALL_VARS}

# Print summaries
for var, res in lmm_results.items():
    print(f"\n{'─'*60}")
    print(f"  {var}")
    print(f"{'─'*60}")
    print(res.summary())


# %% [markdown]
# ## 3 · Pairwise post-hoc (paired t-test, Bonferroni)

def _posthoc(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Paired t-tests across all condition pairs with Bonferroni correction."""
    pivot = df.pivot(index="participant_id", columns="condition", values=dv)
    rows = []
    for c1, c2 in PAIRS:
        if c1 not in pivot.columns or c2 not in pivot.columns:
            continue
        common = pivot[[c1, c2]].dropna()
        t, p = stats.ttest_rel(common[c1], common[c2])
        diff  = common[c1] - common[c2]
        d     = diff.mean() / diff.std()
        rows.append({"A": c1, "B": c2, "t": round(t, 3),
                     "p_raw": round(p, 4), "cohen_d": round(d, 3)})
    res = pd.DataFrame(rows)
    res["p_bonf"] = (res["p_raw"] * len(PAIRS)).clip(upper=1.0).round(4)
    res["sig"]    = res["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01
                  else ("*" if p < 0.05 else "ns"))
    )
    return res

posthoc_all = {var: _posthoc(df, var) for var in ALL_VARS}

# Print summary
for var, ph in posthoc_all.items():
    print(f"\n── {var} ──")
    print(ph[["A", "B", "t", "p_raw", "p_bonf", "sig"]].to_string(index=False))


# %% [markdown]
# ## 3 · Box plots — Total + all subscales

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes_flat = axes.flatten()

for ax, var in zip(axes_flat, ALL_VARS):
    ph = posthoc_all[var]

    sns.boxplot(
        data=df, x="condition", y=var,
        order=ORDER, palette=PALETTE, width=0.55,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        ax=ax,
    )
    sns.stripplot(
        data=df, x="condition", y=var,
        order=ORDER,
        color="black", size=4, alpha=0.5, jitter=True,
        ax=ax,
    )

    sig_pairs  = [(r["A"], r["B"]) for _, r in ph.iterrows()]
    sig_labels = list(ph["sig"])

    annotator = Annotator(ax, sig_pairs, data=df,
                          x="condition", y=var, order=ORDER)
    annotator.configure(line_width=1.0, text_format="simple", fontsize=10)
    annotator.set_custom_annotations(sig_labels)
    annotator.annotate()

    ax.set_title(var, fontsize=11, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–100)", fontsize=9)
    ax.set_ylim(-5, 115)
    ax.tick_params(axis="x", labelsize=9)

    # LMM footnote
    res = lmm_results[var]
    fe, pv = res.fe_params, res.pvalues
    note = "\n".join(
        f"{k.replace(f'C(condition, Treatment(reference={chr(39)}{_REF}{chr(39)}))[T.', '').rstrip(']')}: "
        f"β={v:.1f}, p={pv[k]:.3f}"
        for k, v in fe.items() if k != "Intercept"
    )
    ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=7,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

# Hide the unused 8th subplot (7 vars, 8 slots)
axes_flat[-1].set_visible(False)

fig.suptitle("NASA-TLX by Condition", fontsize=14, y=1.01)
plt.tight_layout()

_plot_file = DATA_ROOT / "tlx_boxplots.png"
plt.savefig(_plot_file, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_plot_file.name}")

# %%
