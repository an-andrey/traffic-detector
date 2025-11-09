import pandas as pd
from ast import literal_eval

# 1) Load
label_cols = [i for i in range(50)]
df = pd.read_csv(
    "Crash-1500.txt",
    names=["vidname"] + label_cols + ["binlabels","startframe","youtubeID","timing","weather","egoinvolve"],
    header=None
)

# 2) Identify frame columns (0..49)
frame_cols = [c for c in df.columns if c != "vidname" and str(c).isdigit()]

# 3) Wide -> long: one row per (video, frame)
long = df.melt(
    id_vars=["vidname"],
    value_vars=frame_cols,
    var_name="frame_idx",
    value_name="crash"
)

# 4) Types
long["frame_idx"] = long["frame_idx"].astype(int)
long["crash"] = pd.to_numeric(long["crash"], errors="coerce").fillna(0).astype(int)

# 5) Sort by numeric video id then frame index
#    Convert vidname to numeric for proper ordering, but keep original for padding
vidnum = pd.to_numeric(long["vidname"], errors="coerce").astype(int)
long = long.assign(_vidnum=vidnum).sort_values(by=["_vidnum", "frame_idx"], kind="mergesort")

# 6) Build desired id: 6-digit padded video id + "-" + unpadded frame index
long["vidname"] = long["_vidnum"].astype(str).str.zfill(6) + "-" + long["frame_idx"].astype(str)


long = long.drop(["frame_idx", "_vidnum"], axis=1)
long = long.reset_index(drop=True)

print(long)

# Optional: save
long.to_csv("dataset.csv", index=False)
