import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader, random_split

def read_data() -> pd.DataFrame:
    # データの読み込み & Salary の欠損行を削除
    hitters_csv = (
        pd.read_csv("Hitters.csv", index_col=None, header=0)
          .dropna(subset=["Salary"])
          .reset_index(drop=True)
    )
    hitters_data = hitters_csv.copy()

    # カテゴリカルデータを数値に変換
    hitters_data["League"]    = hitters_data["League"].map({"A": 0, "N": 1})
    hitters_data["Division"]  = hitters_data["Division"].map({"E": 0, "W": 1})
    hitters_data["NewLeague"] = hitters_data["NewLeague"].map({"A": 0, "N": 1})

    # 数値特徴量をまとめて標準化
    num_cols = hitters_data.drop(columns=["League", "Division", "NewLeague", "Salary"]).columns
    scaler = StandardScaler()
    hitters_data[num_cols] = scaler.fit_transform(hitters_data[num_cols])

    # 目的変数を log1p 変換
    hitters_data["Salary"] = np.log1p(hitters_data["Salary"])

    # 確認
    print(hitters_data.head())
    return hitters_data

def create_dataset_from_dataframe(
    df: pd.DataFrame, target_tag: str = "Salary"
) -> tuple[torch.Tensor, torch.Tensor]:
    # "Salary"の列を目的にする
    target = torch.tensor(df[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "Salary"以外の列を入力にする
    inputs = torch.tensor(df.drop(target_tag, axis=1).values, dtype=torch.float32)
    return inputs, target

class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        return self.l3(h2)

def train_model(nn_model: FourLayerNN, input: torch.Tensor, target: torch.Tensor) -> None:
    # データセットの作成
    dataset = TensorDataset(input, target)
    # train:val = 80:20 に分割
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val   = n - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=25, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=25, shuffle=False)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01, momentum=0.9)

    # データセット全体に対して10000回学習
    for epoch in range(10000):
        nn_model.train()
        for x, y_true in train_loader:
            y_pred = nn_model(x)
            loss   = torch.nn.functional.mse_loss(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 1000回に1回検証データで誤差を表示
        if epoch % 1000 == 0:
            nn_model.eval()
            with torch.inference_mode():
                val_losses = []
                for vx, vy in val_loader:
                    vp = nn_model(vx)
                    val_losses.append(torch.nn.functional.mse_loss(vp, vy).item())
                avg_val = sum(val_losses) / len(val_losses)
                print(f"epoch {epoch:4d}  val MSE: {avg_val:.6f}")

if __name__ == "__main__":
    torch.manual_seed(42)

    # データ準備
    hitters_data = read_data()
    inputs, targets = create_dataset_from_dataframe(hitters_data)

    # モデル生成
    model = FourLayerNN(inputs.shape[1], hidden_size=30, output_size=1)

    # 学習
    train_model(model, inputs, targets)

    # --- テスト評価 ---
    # テストセット分割（デモ用に検証セットと同じ分割を再利用してもOK）
    dataset = TensorDataset(inputs, targets)
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    _, _, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_ds, batch_size=25, shuffle=False)

    # テストMSE/RMSE と R^2 を計算
    model.eval()
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for xb, yb in test_loader:
            yp = model(xb)
            y_trues.extend(yb.cpu().numpy().flatten().tolist())
            y_preds.extend(yp.cpu().numpy().flatten().tolist())

    mse_test  = np.mean((np.array(y_trues) - np.array(y_preds))**2)
    rmse_test = np.sqrt(mse_test)
    r2         = r2_score(y_trues, y_preds)
    print(f"Test MSE: {mse_test:.6f}  RMSE: {rmse_test:.6f}  R^2: {r2:.3f}")

    # サンプル予測
    sample = hitters_data.median().drop("Salary").values
    test_data = torch.tensor([sample], dtype=torch.float32)
    with torch.inference_mode():
        pred_log = model(test_data).item()
    print(f"Sample pred (log1p): {pred_log:.4f}  => Salary: {np.expm1(pred_log):,.0f}")
