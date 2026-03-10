# %% [markdown]
# # Participant ID ↔ Name Mapping
# Extracts participant ID and Korean name pairs from folder names under the project root.
# Folder naming convention: `P{id}_{korean_name}`

# %%
import re
from pathlib import Path

# Resolve project root whether running as a script or in Jupyter
try:
    DATA_ROOT = Path(__file__).parent
except NameError:
    DATA_ROOT = Path().resolve()

participants = []
for folder in DATA_ROOT.iterdir():
    match = re.match(r'^P(\d+)_(.+)$', folder.name)
    if match and folder.is_dir():
        pid = int(match.group(1))
        korean_name = match.group(2)
        participants.append((pid, korean_name))

participants.sort(key=lambda x: x[0])

# %%
import pandas as pd

df_participants = pd.DataFrame(participants, columns=["participant_id", "name"])
df_participants["participant_id"] = df_participants["participant_id"].apply(lambda x: f"P{x}")

# %% [markdown]
# ## Expertise levels
# Word → integer mapping (hardcoded scale).
# Values are parsed from the leading Korean keyword in each cell of the demographics CSV.

USAGE_SCALE = {
    "전혀 없음":    "1",
    "단순 체험":    "2",
    "가끔 사용":    "3",
    "자주 사용":    "4",
    "일상적 사용":  "5",
}
EXPERTISE_SCALE = {
    "전혀 없음":   "1",
    "단순 체험":   "2",
    "기본적 경험": "3",
    "익숙함":      "4",
    "매우 능숙함": "5",
}

_demo = pd.read_csv(DATA_ROOT / "participants_demographics.csv", encoding="utf-8-sig")

# Extract the leading keyword (everything before the first space-paren pattern)
def _extract_keyword(cell: str) -> str:
    return re.split(r'\s*\(', str(cell))[0].strip()

_demo["headset_usage"] = (
    _demo["VR/AR headset usage level"]
    .apply(_extract_keyword)
    .map(USAGE_SCALE)
)
_demo["bare_hand_expertise"] = (
    _demo["Bare-hand interaction experience level"]
    .apply(_extract_keyword)
    .map(EXPERTISE_SCALE)
)
_demo["direct_manipulation_expertise"] = (
    _demo["Direct manipulation experience level"]
    .apply(_extract_keyword)
    .map(EXPERTISE_SCALE)
)

# Merge into df_participants by matching Korean name
df_participants = df_participants.merge(
    _demo[["Name", "headset_usage", "bare_hand_expertise", "direct_manipulation_expertise"]],
    left_on="name",
    right_on="Name",
).drop(columns="Name")

print(f"{len(df_participants)} participants found")
df_participants

# %% [markdown]
# ## Post-experiment survey
# Extracts English-named columns from `participants_postexp.csv`.

_postexp = pd.read_csv(DATA_ROOT / "participants_postexp.csv", encoding="utf-8-sig")

_NAME_COL = "참가자 이름 (데이터 식별을 위함이며 연구에 사용되지 않음)"
_english_cols = [c for c in _postexp.columns if c.isascii() and c.strip()]

df_participants = df_participants.merge(
    _postexp[[_NAME_COL] + _english_cols].rename(columns={_NAME_COL: "name"}),
    on="name",
)

print(f"Post-exp columns added: {_english_cols}")
df_participants

# %% [markdown]
# ## Post-condition questionnaire
# Three rows per participant (one per condition).
# English columns are extracted and expertise columns are joined from `df_participants`.


_postcond = pd.read_csv(DATA_ROOT / "postcondition.csv", encoding="utf-8-sig")

_NAME_COL_PC = "참가자 이름"
_english_cols_pc = [c for c in _postcond.columns if c.isascii() and c.strip()]

df_postcondition = (
    _postcond[[_NAME_COL_PC] + _english_cols_pc]
    .rename(columns={_NAME_COL_PC: "name"})
    .merge(
        df_participants[["name", "participant_id", "headset_usage", "bare_hand_expertise", "direct_manipulation_expertise"]],
        on="name",
    )
)

# Compute VEQ as the mean of the four AG items
_ag_cols = ["AG1", "AG2", "AG3", "AG4"]
df_postcondition["VEQ"] = df_postcondition[_ag_cols].mean(axis=1).round(2)

# Reorder: participant_id first, then expertise columns, then questionnaire columns
_q_cols = [c for c in _english_cols_pc if c != "Condition"]
df_postcondition = df_postcondition[
    ["participant_id", "name", "Condition", "headset_usage", "bare_hand_expertise", "direct_manipulation_expertise",
     "VEQ"] + _q_cols
]

print(f"Post-condition: {df_postcondition.shape[0]} rows × {df_postcondition.shape[1]} cols")
df_postcondition

# %%
