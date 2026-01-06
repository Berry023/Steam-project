import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def num(x):
    return pd.to_numeric(x, errors="coerce")

def cost(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "free" in s:
        return 0.0
    m = re.search(r"(\d+(\.\d+)?)", s.replace(",", "."))
    return float(m.group(1)) if m else np.nan

def gb(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    m = re.search(r"(\d+(\.\d+)?)\s*(tb|gb|mb)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    u = m.group(3)
    if u == "tb":
        v *= 1024
    if u == "mb":
        v /= 1024
    return v

def cnt(s, key):
    if pd.isna(s):
        return 0
    m = re.search(rf"{key}\s*:\s*(\d+)", str(s).lower())
    return int(m.group(1)) if m else 0

def run_models(x, y, logreg_scaled):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    sc = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}
    if logreg_scaled:
        m1 = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
        ])
    else:
        m1 = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])
    m2 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=1,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        ))
    ])
    m3 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=6,
            max_iter=200,
            random_state=42
        ))
    ])

    def one(name, res):
        return (name,
                float(np.mean(res["test_pr_auc"])),
                float(np.mean(res["test_roc_auc"])),
                float(np.mean(res["test_f1"])))

    r1 = cross_validate(m1, x, y, cv=cv, scoring=sc, n_jobs=1)
    r2 = cross_validate(m2, x, y, cv=cv, scoring=sc, n_jobs=1)
    r3 = cross_validate(m3, x, y, cv=cv, scoring=sc, n_jobs=1)
    return [one("LogReg", r1), one("RF", r2), one("HGB", r3)]

def print_res(res):
    for name, pr, roc, f1 in res:
        print(f"  {name}: PR-AUC={pr:.3f}, ROC-AUC={roc:.3f}, F1={f1:.3f}")

def print_one(name, pr, roc, f1):
    print(f"  {name}: PR-AUC={pr:.3f}, ROC-AUC={roc:.3f}, F1={f1:.3f}")

df = pd.read_csv(
    "data.csv",
    sep=";",
    engine="python",
    usecols=[
        "Presence",
        "Metacritic", "Indie", "Soundtrack", "Controller", "Achievements",
        "OriginalCost", "DiscountedCost",
        "ReleaseDate",
        "Platform", "Languages", "Players",
        "RatingsBreakdown",
        "Memory", "Storage",
        "Genres", "Tags", "Description"
    ]
)

p = num(df["Presence"])
thr = p.quantile(0.8)
y = (p >= thr).astype(int)

n = len(df)
pos = int(y.sum())
neg = n - pos
pos_pct = 100.0 * pos / n
print(f"rows={n}, pos={pos} ({pos_pct:.1f}%), neg={neg}")

x1 = df[["Metacritic", "Indie", "Soundtrack", "Controller", "Achievements"]].copy()
for c in x1.columns:
    x1[c] = num(x1[c])

print(f"\ntry1: features={x1.shape[1]}")
print("running CV...")
res1 = run_models(x1, y, logreg_scaled=False)
print("results:")
print_res(res1)

x2 = pd.DataFrame()
x2["met"] = num(df["Metacritic"])
x2["indie"] = num(df["Indie"])
x2["sound"] = num(df["Soundtrack"])
x2["ctrl"] = num(df["Controller"])
x2["ach"] = num(df["Achievements"])
x2["orig"] = df["OriginalCost"].map(cost)
x2["disc"] = df["DiscountedCost"].map(cost)
dt = pd.to_datetime(df["ReleaseDate"], errors="coerce")
ref = pd.Timestamp("2020-07-01")
x2["days"] = (ref - dt).dt.days
x2["year"] = dt.dt.year
x2["mem_gb"] = df["Memory"].map(gb)
x2["sto_gb"] = df["Storage"].map(gb)
pl = df["Platform"].fillna("").astype(str).str.lower()
x2["pc"] = pl.str.contains("pc", regex=False).astype(int)
x2["mac"] = pl.str.contains("mac", regex=False).astype(int)
x2["linux"] = pl.str.contains("linux", regex=False).astype(int)
x2["xbox"] = pl.str.contains("xbox", regex=False).astype(int)
x2["ps"] = pl.str.contains("playstation", regex=False).astype(int)
x2["nin"] = pl.str.contains("nintendo", regex=False).astype(int)
lang = df["Languages"].fillna("").astype(str)
x2["lang_n"] = lang.map(lambda s: len([z for z in s.split(",") if z.strip()]))
plr = df["Players"].fillna("").astype(str).str.lower()
x2["multi"] = plr.str.contains("multiplayer", regex=False).astype(int)
x2["single"] = plr.str.contains("singleplayer", regex=False).astype(int)
x2["coop"] = plr.str.contains("coop", regex=False).astype(int)
x2["pvp"] = plr.str.contains("pvp", regex=False).astype(int)
rb = df["RatingsBreakdown"]
x2["rec"] = rb.map(lambda s: cnt(s, "recommended"))
x2["meh"] = rb.map(lambda s: cnt(s, "meh"))
x2["exc"] = rb.map(lambda s: cnt(s, "exceptional"))
x2["skip"] = rb.map(lambda s: cnt(s, "skip"))
x2["tot"] = x2["rec"] + x2["meh"] + x2["exc"] + x2["skip"]
print(f"\ntry2: features={x2.shape[1]}")
print("running CV...")
res2 = run_models(x2, y, logreg_scaled=True)
print("results:")
print_res(res2)

text = (
    df["Genres"].fillna("").astype(str) + " " +
    df["Tags"].fillna("").astype(str) + " " +
    df["Description"].fillna("").astype(str)
)

z = x2.copy()
z["text"] = text
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
sc = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}
model_tab = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
])
prep = ColumnTransformer([
    ("tab", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), list(x2.columns)),
    ("txt", TfidfVectorizer(max_features=20000, ngram_range=(1, 2)), "text"),
])
model_txt = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
])
print("\ntry3A: tabular only (LogReg)")
rA = cross_validate(model_tab, x2, y, cv=cv, scoring=sc, n_jobs=1)
print_one("LogReg", float(np.mean(rA["test_pr_auc"])), float(np.mean(rA["test_roc_auc"])), float(np.mean(rA["test_f1"])))
print("\ntry3B: tabular + text (TF-IDF) (LogReg)")
rB = cross_validate(model_txt, z, y, cv=cv, scoring=sc, n_jobs=1)
print_one("LogReg", float(np.mean(rB["test_pr_auc"])), float(np.mean(rB["test_roc_auc"])), float(np.mean(rB["test_f1"])))

print("\ntry4: feature importance (RF)")
X_train, X_test, y_train, y_test = train_test_split(
    x2, y, test_size=0.25, random_state=42, stratify=y
)
model = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=1,
        class_weight="balanced_subsample",
        min_samples_leaf=2
    ))
])
print("training RF...")
model.fit(X_train, y_train)
print("computing importance...")
imp = permutation_importance(
    model, X_test, y_test,
    n_repeats=8,
    random_state=42,
    scoring="average_precision",
    n_jobs=1
)
res = pd.DataFrame({
    "feature": x2.columns,
    "importance_mean": imp.importances_mean,
    "importance_std": imp.importances_std
}).sort_values("importance_mean", ascending=False)
print("top 10:")
for _, r in res.head(10).iterrows():
    print(f"  {r['feature']}: {r['importance_mean']:.4f}")
