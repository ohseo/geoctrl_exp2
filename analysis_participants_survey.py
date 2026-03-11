# %% [markdown]
# # Post-Experiment Survey Analysis
# Relative frequency bar charts for the last four columns of extracted_participants.csv:
# Most Accurate, Least Accurate, Most Preferred, Least Preferred

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_IN_JUPYTER = "ipykernel" in sys.modules

DATA_ROOT = Path(__file__).parent if "__file__" in dir() else Path().resolve()

# %% [markdown]
# ## Load & clean data

df = pd.read_csv(DATA_ROOT / "extracted_participants_wrist.csv", encoding="utf-8-sig")

SURVEY_COLS = ["Most Accurate", "Least Accurate", "Most Preferred", "Least Preferred"]

def _normalize(val: str) -> str:
    """Normalise a response to X, Y, Z, or etc (ambiguous/multi-axis)."""
    s = str(val).strip().upper()
    found = [a for a in ("X", "Y", "Z") if a in s]
    if len(found) > 1:
        return "etc"
    if len(found) == 1:
        return found[0]
    return None  # unknown

for col in SURVEY_COLS:
    df[col] = df[col].apply(_normalize)

print(df[["participant_id"] + SURVEY_COLS])

# %% [markdown]
# ## Compute relative frequencies

AXIS_TO_COND = {"X": "Baseline", "Y": "Physics", "Z": "GeoCtrl", "etc": "etc"}
ORDER  = ["Baseline", "GeoCtrl", "Physics", "etc"]
COLORS = {"Baseline": "#4C72B0", "GeoCtrl": "#55A868", "Physics": "#DD8452", "etc": "#AAAAAA"}

# Remap raw X/Y/Z/etc responses to condition names
for col in SURVEY_COLS:
    df[col] = df[col].map(AXIS_TO_COND)

freq = {}
for col in SURVEY_COLS:
    counts = df[col].value_counts()
    total  = counts.sum()
    freq[col] = {cond: counts.get(cond, 0) / total for cond in ORDER}

freq_df = pd.DataFrame(freq, index=ORDER).T   # rows=questions, cols=conditions
print("\nRelative frequencies:")
print(freq_df.round(3))

# %% [markdown]
# ## Plot — 100% stacked bar chart (one full bar per question)

fig, ax = plt.subplots(figsize=(9, 5))

bar_w = 0.5
x = np.arange(len(SURVEY_COLS))

bottoms = np.zeros(len(SURVEY_COLS))
for cond in ORDER:
    values = np.array([freq[col][cond] for col in SURVEY_COLS])
    bars = ax.bar(x, values, width=bar_w, bottom=bottoms,
                  color=COLORS[cond], label=cond, edgecolor="white", linewidth=0.8)

    # Annotate segments that are wide enough to label
    for i, (bar, v) in enumerate(zip(bars, values)):
        if v >= 0.06:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + v / 2,
                    f"{v:.0%}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white")
    bottoms += values

ax.set_xticks(x)
ax.set_xticklabels(SURVEY_COLS, fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Relative Frequency", fontsize=10)
ax.set_xlabel("Question", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)

fig.suptitle("Post-Experiment Survey: Preferences", fontsize=13)
plt.tight_layout()

_out = DATA_ROOT / "participants_survey_barplot.png"
plt.savefig(_out, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"\nSaved → {_out.name}")

# %% [markdown]
# ## Plot — 2×2 stacked bar chart by direct_manipulation_expertise

EXPERTISE_LEVELS = sorted(df["direct_manipulation_expertise"].dropna().unique())

def _stacked_by_expertise(ax, col):
    """Draw a 100% stacked bar for one survey question, x = expertise level."""
    x = np.arange(len(EXPERTISE_LEVELS))
    bottoms = np.zeros(len(EXPERTISE_LEVELS))

    for cond in ORDER:
        values = np.array([
            df[df["direct_manipulation_expertise"] == lvl][col]
            .value_counts().get(cond, 0)
            / max(df[df["direct_manipulation_expertise"] == lvl][col].count(), 1)
            for lvl in EXPERTISE_LEVELS
        ])
        bars = ax.bar(x, values, width=0.6, bottom=bottoms,
                      color=COLORS[cond], label=cond,
                      edgecolor="white", linewidth=0.8)
        for i, (bar, v) in enumerate(zip(bars, values)):
            if v >= 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + v / 2,
                        f"{v:.0%}",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white")
        bottoms += values

    # Annotate n per expertise level below x-axis
    for i, lvl in enumerate(EXPERTISE_LEVELS):
        n = df[df["direct_manipulation_expertise"] == lvl][col].count()
        ax.text(i, -0.06, f"n={n}", ha="center", va="top", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(lvl)) for lvl in EXPERTISE_LEVELS], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Direct Manipulation Expertise", fontsize=9)
    ax.set_title(col, fontsize=11, pad=6)
    ax.spines[["top", "right"]].set_visible(False)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes2_flat = axes2.flatten()

for ax, col in zip(axes2_flat, SURVEY_COLS):
    _stacked_by_expertise(ax, col)
    ax.set_ylabel("Relative Frequency", fontsize=9)

# Single shared legend
handles, labels = axes2_flat[0].get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    seen.setdefault(l, h)
fig2.legend(seen.values(), seen.keys(),
            title="Condition", bbox_to_anchor=(1.01, 0.5),
            loc="center left", fontsize=10)

fig2.suptitle("Survey Answers by Direct Manipulation Expertise", fontsize=13)
plt.tight_layout()

_out2 = DATA_ROOT / "participants_survey_by_expertise.png"
plt.savefig(_out2, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_out2.name}")

# %% [markdown]
# ## Plot — 2×2 heatmap by direct_manipulation_expertise

fig3, axes3 = plt.subplots(2, 2, figsize=(10, 7))
axes3_flat = axes3.flatten()

for ax, col in zip(axes3_flat, SURVEY_COLS):
    # Build matrix: rows=expertise levels, cols=conditions
    matrix = np.array([
        [
            df[df["direct_manipulation_expertise"] == lvl][col]
            .value_counts().get(cond, 0)
            / max(df[df["direct_manipulation_expertise"] == lvl][col].count(), 1)
            for cond in ORDER
        ]
        for lvl in EXPERTISE_LEVELS
    ])

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    # Annotate cells
    for r in range(len(EXPERTISE_LEVELS)):
        for c in range(len(ORDER)):
            v = matrix[r, c]
            text_color = "white" if v > 0.55 else "black"
            n = df[df["direct_manipulation_expertise"] == EXPERTISE_LEVELS[r]][col].count()
            ax.text(c, r, f"{v:.0%}\n(n={n})",
                    ha="center", va="center", fontsize=9,
                    color=text_color)

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=10)
    ax.set_yticks(range(len(EXPERTISE_LEVELS)))
    ax.set_yticklabels([str(int(lvl)) for lvl in EXPERTISE_LEVELS], fontsize=10)
    ax.set_xlabel("Condition", fontsize=9)
    ax.set_ylabel("Direct Manipulation Expertise", fontsize=9)
    ax.set_title(col, fontsize=11, pad=6)
    fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="{x:.0%}")

fig3.suptitle("Survey Answers by Direct Manipulation Expertise (Heatmap)", fontsize=13)
plt.tight_layout()

_out3 = DATA_ROOT / "participants_survey_heatmap.png"
plt.savefig(_out3, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_out3.name}")

# %% [markdown]
# ## Plot — Wrist Usage Ratio vs Direct Manipulation Expertise

df_wrist = df.dropna(subset=["mean_wrist_usage_ratio", "direct_manipulation_expertise"]).copy()
df_wrist["direct_manipulation_expertise"] = pd.to_numeric(df_wrist["direct_manipulation_expertise"], errors="coerce")
df_wrist["mean_wrist_usage_ratio"] = pd.to_numeric(df_wrist["mean_wrist_usage_ratio"], errors="coerce")
df_wrist = df_wrist.dropna(subset=["mean_wrist_usage_ratio", "direct_manipulation_expertise"])

fig4, ax4 = plt.subplots(figsize=(7, 5))

expertise_levels = sorted(df_wrist["direct_manipulation_expertise"].unique())
jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(df_wrist))

ax4.scatter(
    df_wrist["direct_manipulation_expertise"] + jitter,
    df_wrist["mean_wrist_usage_ratio"],
    s=60, alpha=0.7, color="#55A868", edgecolors="white", linewidths=0.6,
)

means = df_wrist.groupby("direct_manipulation_expertise")["mean_wrist_usage_ratio"].mean()
ax4.plot(means.index, means.values, color="#333333", linewidth=1.5,
         marker="D", markersize=6, zorder=5, label="Group mean")

ax4.set_xticks(expertise_levels)
ax4.set_xticklabels([str(int(lvl)) for lvl in expertise_levels], fontsize=11)
ax4.set_xlabel("Direct Manipulation Expertise", fontsize=11)
ax4.set_ylabel("Mean Wrist Usage Ratio (GeoCtrl)", fontsize=11)
ax4.set_title("Wrist Usage Ratio by Direct Manipulation Expertise", fontsize=12)
ax4.legend(fontsize=10)
ax4.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

_out4 = DATA_ROOT / "wrist_usage_by_expertise.png"
plt.savefig(_out4, dpi=150, bbox_inches="tight")
if _IN_JUPYTER:
    plt.show()
else:
    plt.close()
print(f"Saved → {_out4.name}")

# %%
