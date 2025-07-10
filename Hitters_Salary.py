#!/usr/bin/env python
"""
Hitters 年俸予測 + SHAP 可視化
──────────────────────────────────────────
Step A : 学習 (EarlyStopping, RMSE 表示, 重み保存)
Step B : ランダム選手データを生成して年俸を予測
Step C : SHAP (DeepExplainer) で特徴寄与度を計算・PNG 保存
生成ファイル: best_model.pth / preproc.joblib / shap_summary.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, shap, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

# ───────────────────────────── 0. パス
DATA_PATH = Path(__file__).parent / "Hitters.csv"

# ───────────────────────────── 1. データ & 前処理
df = pd.read_csv(DATA_PATH).dropna(subset=["Salary"])
y_log = np.log1p(df["Salary"].astype(float).values)
X = df.drop("Salary", axis=1)

cat_cols = ["League", "Division", "NewLeague"]
num_cols = X.columns.difference(cat_cols)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(drop="first", sparse_output=False))
])
preproc = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
X_train = preproc.fit_transform(X_train)
X_test  = preproc.transform(X_test)

train_X = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
test_X  = torch.tensor(X_test,  dtype=torch.float32)
test_y  = torch.tensor(y_test.reshape(-1, 1),  dtype=torch.float32)

loader = DataLoader(TensorDataset(train_X, train_y),
                    batch_size=32, shuffle=True)

# ───────────────────────────── 2. NN
class SalaryNet(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64),  nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),   nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

model = SalaryNet(train_X.shape[1])
opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_f = nn.MSELoss()

# ───────────────────────────── 3. 学習 + EarlyStopping
best_usd, patience, counter = np.inf, 25, 0
EPOCHS = 400
for epoch in trange(EPOCHS, desc="epochs"):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_f(model(xb), yb).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_log = model(test_X)
        y_np, p_np = test_y.numpy(), pred_log.numpy()
        rmse_log = root_mean_squared_error(y_np, p_np)
        rmse_usd = root_mean_squared_error(np.expm1(y_np), np.expm1(p_np))

    if rmse_usd < best_usd - 1e-3:
        best_usd, counter = rmse_usd, 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopped @ epoch {epoch}")
            break

print(f"Best RMSE (log scale) : {rmse_log:.3f}")
print(f"Best RMSE (USD)      : {best_usd:,.0f} 千ドル")
joblib.dump(preproc, "preproc.joblib")
print("Artifacts saved → best_model.pth / preproc.joblib")

# ───────────────────────────── 4. ランダム選手で推論
def random_player():
    return {
        "AtBat": np.random.randint(100, 600),
        "Hits":  np.random.randint(20, 200),
        "HmRun": np.random.randint(0, 40),
        "Runs":  np.random.randint(10, 120),
        "RBI":   np.random.randint(10, 120),
        "Walks": np.random.randint(0, 100),
        "Years": np.random.randint(1, 15),
        "CAtBat": np.random.randint(100, 6000),
        "CHits":  np.random.randint(50, 2000),
        "CHmRun": np.random.randint(0, 300),
        "CRuns":  np.random.randint(30, 1200),
        "CRBI":   np.random.randint(20, 1200),
        "CWalks": np.random.randint(10, 1200),
        "PutOuts": np.random.randint(0, 1400),
        "Assists": np.random.randint(0, 500),
        "Errors":  np.random.randint(0, 30),
        "League":    np.random.choice(["A", "N"]),
        "Division":  np.random.choice(["E", "W"]),
        "NewLeague": np.random.choice(["A", "N"])
    }

sample_df = pd.DataFrame([random_player()])
X_new = torch.tensor(preproc.transform(sample_df), dtype=torch.float32)
with torch.no_grad():
    pred_k = np.expm1(model(X_new).item())

print("\n=== ランダム選手 1986 成績 ===")
print(sample_df.to_string(index=False))
print(f"\n⇒ 予測年俸 : 約 {pred_k:,.0f} 千ドル")

# ───────────────────────────── 5. SHAP 可視化
print("\nCalculating SHAP values with unified Explainer (may take ~3-4 min)…")

# 1) 予測関数: DataFrame → ndarray
def model_predict(df: pd.DataFrame) -> np.ndarray:
    with torch.inference_mode():
        t = torch.tensor(df.values, dtype=torch.float32)
        return model(t).cpu().numpy()

# 2) 背景 / テスト DataFrame （列名付き）
feature_names = [n.split("__")[-1] for n in preproc.get_feature_names_out()]
background_df = pd.DataFrame(train_X.numpy(), columns=feature_names).sample(200, random_state=0)
test_df       = pd.DataFrame(test_X.numpy(),   columns=feature_names)

import shap
explainer   = shap.Explainer(model_predict, background_df, feature_names=feature_names)
shap_values = explainer(test_df, silent=True)          # shap_values: shap.Explanation

# 3) 上位 20 特徴のバー & summary プロット
import matplotlib.pyplot as plt
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=200)

shap.summary_plot(shap_values, max_display=20, show=False)  # beeswarm
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=200)
plt.close("all")

print("✅ SHAP figures saved → shap_bar.png / shap_beeswarm.png")