# %% [markdown]
# # Post-Condition Questionnaire Analysis
# Friedman test + box plots with significance annotations for Likert-scale items.
# - **Friedman:** non-parametric repeated-measures test across 3 conditions
# - **Post-hoc:** Wilcoxon signed-rank tests with Bonferroni correction

# %%
import sys
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statannotations.Annotator import Annotator
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*reconfigured without applying.*")

_IN_JUPYTER = "ipykernel" in sys.modules

# ── load data if not already in scope ─────────────────────────────────────────
try:
    df_postcondition
    df_participants
    DATA_ROOT
except NameError:
    import runpy, pathlib
    _ns = runpy.run_path(str(pathlib.Path(__file__).parent / "extract_participants.py"))
    df_postcondition = _ns["df_postcondition"]
    df_participants  = _ns["df_participants"]
    DATA_ROOT        = _ns["DATA_ROOT"]


# %% [markdown]
# ## Configuration

CONDITION_MAP = {"X": "Baseline", "Y": "Physics", "Z": "GeoCtrl"}
ORDER   = ["Baseline", "GeoCtrl", "Physics"]
PALETTE = {"Baseline": "#4C72B0", "Physics": "#DD8452", "GeoCtrl": "#55A868"}
PILOTS  = ["P0", "P99", "P26", "P30"]
SHOW_NS = False   # True = annotate all pairs including ns; False = sig pairs only

LIKERT_VARS = ["AG1", "AG2", "AG3", "AG4", "VEQ", "expected", "realistic", "comfortable", "easy", "learnable"]

PAIRS = list(combinations(ORDER, 2))


# %% [markdown]
# ## 1 · Prepare data

df = df_postcondition[~df_postcondition["participant_id"].isin(PILOTS)].copy()
df["condition"] = df["Condition"].map(CONDITION_MAP)
df["condition"] = pd.Categorical(df["condition"], categories=ORDER, ordered=True)

for col in LIKERT_VARS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Participants: {sorted(df['participant_id'].unique())}")
print(f"N = {df['participant_id'].nunique()}")
print()
print(df.groupby("condition")[LIKERT_VARS].median().round(2))
print(df.groupby("condition")[LIKERT_VARS].mean())


# %% [markdown]
# ## 2 · Friedman test (per variable)

def _run_friedman(df: pd.DataFrame, dv: str) -> dict:
    """Friedman test across the three conditions for one dependent variable."""
    pivot = df.pivot_table(index="participant_id", columns="condition", values=dv)
    pivot = pivot[ORDER].dropna()
    stat, p = stats.friedmanchisquare(*[pivot[c] for c in ORDER])
    return {"chi2": round(stat, 3), "p": round(p, 4), "N": len(pivot)}

friedman_results = {var: _run_friedman(df, var) for var in LIKERT_VARS}

print(f"\n{'Variable':<25} {'χ²':>7} {'p':>8} {'N':>4}")
print("─" * 48)
for var, r in friedman_results.items():
    sig = " *" if r["p"] < 0.05 else ""
    print(f"{var:<25} {r['chi2']:>7.3f} {r['p']:>8.4f} {r['N']:>4}{sig}")


# %% [markdown]
# ## 3 · Pairwise post-hoc (Wilcoxon signed-rank, Bonferroni)

def _posthoc(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Wilcoxon signed-rank tests across all condition pairs with Bonferroni correction."""
    pivot = df.pivot_table(index="participant_id", columns="condition", values=dv)
    rows = []
    for c1, c2 in PAIRS:
        if c1 not in pivot.columns or c2 not in pivot.columns:
            continue
        common = pivot[[c1, c2]].dropna()
        stat, p = stats.wilcoxon(common[c1], common[c2])
        rows.append({"A": c1, "B": c2, "W": round(stat, 3), "p_raw": round(p, 4)})
    res = pd.DataFrame(rows)
    res["p_bonf"] = (res["p_raw"] * len(PAIRS)).clip(upper=1.0).round(4)
    res["sig"]    = res["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01
                  else ("*" if p < 0.05 else "ns"))
    )
    return res

posthoc_all = {var: _posthoc(df, var) for var in LIKERT_VARS}

for var, ph in posthoc_all.items():
    print(f"\n── {var} ──")
    print(ph[["A", "B", "W", "p_raw", "p_bonf", "sig"]].to_string(index=False))


# %% [markdown]
# ## 4 · Box plots — all Likert items

n_vars = len(LIKERT_VARS)
n_cols = 5
n_rows = (n_vars + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes_flat = axes.flatten()

for ax, var in zip(axes_flat, LIKERT_VARS):
    ph = posthoc_all[var]
    fr = friedman_results[var]

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

    _ph_plot   = ph if SHOW_NS else ph[ph["sig"] != "ns"]
    sig_pairs  = [(r["A"], r["B"]) for _, r in _ph_plot.iterrows()]
    if sig_pairs:
        annotator = Annotator(ax, sig_pairs, data=df,
                              x="condition", y=var, order=ORDER)
        annotator.configure(line_width=1.0, text_format="simple", fontsize=10)
        annotator.set_custom_annotations(list(_ph_plot["sig"]))
        annotator.annotate()

    ax.set_title(var, fontsize=11, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("Likert score", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)

    # Friedman footnote
    note = f"Friedman: χ²={fr['chi2']:.3f}, p={fr['p']:.4f}"
    ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=7,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))

for ax in axes_flat[n_vars:]:
    ax.set_visible(False)

fig.suptitle("Post-Condition Questionnaire by Condition", fontsize=14, y=1.01)
plt.tight_layout()

_plot_file = DATA_ROOT / "postcondition_boxplots.png"
plt.savefig(_plot_file, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_plot_file.name}")

# %%
