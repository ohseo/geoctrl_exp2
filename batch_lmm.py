"""
Batch LMM runner
────────────────
Runs the full LMM pipeline (descriptives → LMM → post-hoc → box plot)
for every DV listed in BATCH_DVS, saves one PNG per DV, and writes all
descriptive stats + LMM summaries to a single text report.

Usage:
    python batch_lmm.py
"""

import io
import pathlib
import runpy
import sys
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statannotations.Annotator import Annotator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*reconfigured without applying.*")
warnings.filterwarnings("ignore", message=".*Random effects covariance is singular.*")

_HERE = pathlib.Path(__file__).parent

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
_ns = runpy.run_path(str(_HERE / "load_data.py"))
df_summary = _ns["df_summary"]
DATA_ROOT  = _ns["DATA_ROOT"]

_ep_ns = runpy.run_path(str(_HERE / "extract_participants.py"))
df_participants = _ep_ns["df_participants"]

# Merge expertise columns if missing
_EXPERTISE_COLS = ["headset_usage", "bare_hand_expertise", "direct_manipulation_expertise"]
if not all(c in df_summary.columns for c in _EXPERTISE_COLS):
    _df_p = df_participants[["participant_id"] + _EXPERTISE_COLS].copy()
    for c in _EXPERTISE_COLS:
        _df_p[c] = pd.to_numeric(_df_p[c], errors="coerce")
    df_summary = df_summary.merge(_df_p, on="participant_id", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here
# ══════════════════════════════════════════════════════════════════════════════

BATCH_DVS = [
    "Task Completion Time",
    "Total Die Rotation",
    "Total Die Translation",
    "Sum Wrist Rotation",
    "Sum Wrist Translation",
    "Total Head Rotation",
    "Total Head Translation",
    "Target Offset Rotation Angle",
    "Target Offset Position Magnitude",
]

EXCLUDE_TIMEOUTS       = False
EXCLUDE_OUTLIERS       = False
EXCLUDE_NON_EXPERTS    = False
EXCLUDE_TRIAL_OUTLIERS = False   # remove individual trials whose DV is an outlier
SHOW_NS                = False   # True = annotate all pairs including ns; False = sig pairs only
TRIAL_OUTLIER_METHOD   = "iqr"   # "iqr" or "zscore"
TRIAL_OUTLIER_THRESHOLD = 1.5    # IQR multiplier (1.5) or z-score cutoff (e.g. 2.5)

IV1       = "condition"
IV1_ORDER = ["Baseline", "GeoCtrl", "Physics"]
IV2       = None   # set to e.g. "Block Num" to add interaction term

ORDER   = ["Baseline", "GeoCtrl", "Physics"]
PALETTE = {"Baseline": "#4C72B0", "Physics": "#DD8452", "GeoCtrl": "#55A868"}

VALUE_OUTLIERS = ["P23", "P18", "P4", "P14"]
NON_EXPERTS = df_participants.loc[
    df_participants["direct_manipulation_expertise"].astype(float) < 3,
    "participant_id",
].tolist()

if EXCLUDE_OUTLIERS:
    OUTLIERS = VALUE_OUTLIERS[:]
else:
    OUTLIERS = []
if EXCLUDE_NON_EXPERTS:
    OUTLIERS = OUTLIERS + NON_EXPERTS

OUTPUT_DIR  = DATA_ROOT / f"lmm_T-{EXCLUDE_TIMEOUTS}_O-{EXCLUDE_OUTLIERS}_N-{EXCLUDE_NON_EXPERTS}_I-{EXCLUDE_TRIAL_OUTLIERS}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "lmm_batch_report.txt"

# ══════════════════════════════════════════════════════════════════════════════


def run_one(dv: str, df_summary: pd.DataFrame, report_lines: list[str]) -> None:
    """Run the full LMM pipeline for a single DV and append results to report_lines."""

    if dv not in df_summary.columns:
        report_lines.append(f"\n{'═'*70}\n  SKIPPED: '{dv}' — column not found\n{'═'*70}\n")
        print(f"  SKIP  '{dv}' — column not found")
        return

    print(f"  RUN   '{dv}' …", end=" ", flush=True)

    _ref       = IV1_ORDER[0]
    _iv1_term  = IV1 if " " not in IV1 else f"Q('{IV1}')"
    _cond_term = f"C({_iv1_term}, Treatment(reference='{_ref}'))"
    _palette   = PALETTE if IV1 == "condition" else "Set2"
    _formula   = (
        f"Q('{dv}') ~ {_cond_term} * C(Q('{IV2}'))"
        if IV2 else
        f"Q('{dv}') ~ {_cond_term}"
    )
    _safe_dv   = dv.replace(" ", "_").replace("/", "_")
    _plot_file = OUTPUT_DIR / f"lmm_{_safe_dv}_T-{EXCLUDE_TIMEOUTS}_O-{EXCLUDE_OUTLIERS}_N-{EXCLUDE_NON_EXPERTS}_I-{EXCLUDE_TRIAL_OUTLIERS}.png"

    # ── data prep ─────────────────────────────────────────────────────────────
    df = df_summary[["participant_id", "condition", "Block Num", "Trial Num",
                      "Is Timeout", dv]].copy()
    df[dv]           = pd.to_numeric(df[dv], errors="coerce")
    df["Is Timeout"] = df["Is Timeout"].astype(str).str.lower() == "true"
    df[IV1]          = pd.Categorical(df[IV1], categories=IV1_ORDER, ordered=True)

    df_clean = df[~df["Is Timeout"]].copy() if EXCLUDE_TIMEOUTS else df.copy()
    df_lmm   = df_clean[~df_clean["participant_id"].isin(OUTLIERS)].copy()

    # ── trial-level outlier removal ───────────────────────────────────────────
    if EXCLUDE_TRIAL_OUTLIERS:
        def _trial_outlier_mask(group: pd.Series) -> pd.Series:
            if TRIAL_OUTLIER_METHOD == "zscore":
                z = (group - group.mean()) / group.std()
                return z.abs() > TRIAL_OUTLIER_THRESHOLD
            else:  # iqr
                q1, q3 = group.quantile(0.25), group.quantile(0.75)
                iqr = q3 - q1
                return (group < q1 - TRIAL_OUTLIER_THRESHOLD * iqr) | \
                       (group > q3 + TRIAL_OUTLIER_THRESHOLD * iqr)

        outlier_flags = df_lmm.groupby(IV1, observed=True)[dv].transform(_trial_outlier_mask)
        n_before = len(df_lmm)
        df_lmm = df_lmm[~outlier_flags].copy()
        print(f"    trial outliers removed: {n_before - len(df_lmm)} "
              f"({TRIAL_OUTLIER_METHOD}, threshold={TRIAL_OUTLIER_THRESHOLD})", end=" ")

    # ── descriptives ──────────────────────────────────────────────────────────
    desc = (
        df_lmm.groupby(IV1, observed=True)[dv]
        .agg(n="count", mean="mean", sd="std", median="median",
             q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
        .round(3)
    )

    # ── LMM ───────────────────────────────────────────────────────────────────
    model  = smf.mixedlm(_formula, data=df_lmm, groups=df_lmm["participant_id"])
    result = model.fit(method="lbfgs", reml=True)

    # ── post-hoc (ttest_ind) ──────────────────────────────────────────────────
    pairs = list(combinations(IV1_ORDER, 2))
    posthoc_rows = []
    for c1, c2 in pairs:
        g1 = df_lmm.loc[df_lmm[IV1] == c1, dv]
        g2 = df_lmm.loc[df_lmm[IV1] == c2, dv]
        t_stat, p_raw = stats.ttest_ind(g1, g2)
        denom = np.sqrt(
            ((len(g1) - 1) * g1.std()**2 + (len(g2) - 1) * g2.std()**2)
            / (len(g1) + len(g2) - 2)
        )
        cohens_d = (g1.mean() - g2.mean()) / denom if denom else np.nan
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

    # ── box plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_lmm, x=IV1, y=dv, order=IV1_ORDER, palette=_palette,
                width=0.5, flierprops=dict(marker="o", markersize=4, alpha=0.5), ax=ax)
    # sns.stripplot(data=df_lmm, x=IV1, y=dv, order=IV1_ORDER,
    #               color="black", size=3, alpha=0.35, jitter=True, ax=ax)

    _ph_plot   = df_posthoc if SHOW_NS else df_posthoc[df_posthoc["sig"] != "ns"]
    sig_pairs  = [(r["pair"].split(" vs ")[0], r["pair"].split(" vs ")[1])
                  for _, r in _ph_plot.iterrows()]
    if sig_pairs:
        annotator = Annotator(ax, sig_pairs, data=df_lmm, x=IV1, y=dv, order=IV1_ORDER)
        annotator.configure(line_width=1.2, text_format="simple", fontsize=11)
        annotator.set_custom_annotations(list(_ph_plot["sig"]))
        annotator.annotate()

    ax.set_title(f"{dv} by {IV1}", fontsize=13, pad=12)
    ax.set_xlabel(IV1, fontsize=11)
    ax.set_ylabel(dv, fontsize=11)
    ax.tick_params(axis="x", labelsize=11)

    _stats_text = (
        f"  |  ".join(
            f"{cond}: {desc['mean'][cond]:.2f} (SD={desc['sd'][cond]:.2f})"
            for cond in IV1_ORDER if cond in desc.index
        ) + f"  |  T={EXCLUDE_TIMEOUTS} O={EXCLUDE_OUTLIERS} N={EXCLUDE_NON_EXPERTS} I={EXCLUDE_TRIAL_OUTLIERS}"
    )
    fig.text(0.5, -0.02, _stats_text, ha="center", va="top",
             fontsize=6.5, color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(_plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    # ── append to report ──────────────────────────────────────────────────────
    lmm_buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = lmm_buf
    print(result.summary())
    sys.stdout = sys_stdout

    icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)

    report_lines += [
        f"\n{'═'*70}",
        f"  DV: {dv}",
        f"  Formula: {_formula}",
        f"  Outliers excluded: {OUTLIERS}",
        f"  Timeouts excluded: {EXCLUDE_TIMEOUTS}",
        f"{'═'*70}",
        "",
        "── Descriptive statistics ──────────────────────────────────────────",
        desc.to_string(),
        "",
        "── LMM summary ─────────────────────────────────────────────────────",
        lmm_buf.getvalue().strip(),
        f"\n  ICC = {icc:.3f}",
        "",
        "── Post-hoc (Bonferroni-corrected t-tests) ─────────────────────────",
        df_posthoc.to_string(index=False),
        "",
    ]

    print(f"done → {_plot_file.name}")


# ── batch loop ────────────────────────────────────────────────────────────────
report_lines = [
    "LMM Batch Report",
    f"IV1={IV1}  IV1_ORDER={IV1_ORDER}  IV2={IV2}",
    f"EXCLUDE_TIMEOUTS={EXCLUDE_TIMEOUTS}  EXCLUDE_OUTLIERS={EXCLUDE_OUTLIERS}  "
    f"EXCLUDE_NON_EXPERTS={EXCLUDE_NON_EXPERTS} EXCLUDE_TRIAL_OUTLIERS={EXCLUDE_TRIAL_OUTLIERS}",
    f"OUTLIERS={OUTLIERS}",
]

print(f"\nRunning batch for {len(BATCH_DVS)} DVs …")
for dv in BATCH_DVS:
    run_one(dv, df_summary, report_lines)

REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\nDone. Report saved → {REPORT_FILE.name}")
