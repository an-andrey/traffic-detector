import pandas as pd
from ast import literal_eval

# 1) Load
label_cols = [i for i in range(50)]
df = pd.read_csv("Crash-1500.txt", names=["vidname"] + label_cols + ["binlabels","startframe","youtubeID","timing","weather","egoinvolve"], header=None)

# print(df)

df.to_csv("dataset.csv")