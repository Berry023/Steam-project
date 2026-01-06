import io
import csv
import numpy as np
import pandas as pd

a = []
d = []
in_data = False

with open("dataset_", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if not in_data:
            if s.lower().startswith("@attribute"):
                a.append(s.split(None, 2)[1])
            elif s.lower() == "@data":
                in_data = True
        else:
            d.append(line.rstrip("\n"))

r = csv.reader(io.StringIO("\n".join(d)), delimiter=",", quotechar="'", escapechar="\\")
rows = list(r)
df = pd.DataFrame(rows, columns=a).replace("?", np.nan)

df.to_csv("data.csv", index=False, sep=";")
print("saved data.csv")
