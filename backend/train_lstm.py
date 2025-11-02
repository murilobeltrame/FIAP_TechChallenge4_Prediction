# backend/train_lstm.py
import os
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime  # ✅ adicionado para log

var_input_size = 5
var_hidden_size = 128
var_num_layers = 2
var_output_size = 1
var_dropout = 0.2

# ============================================================
# MODELO LSTM — múltiplas features (Open, High, Low, Close, Volume)
# ============================================================
class StockLSTM(nn.Module):
    def __init__(self, input_size=var_input_size, hidden_size=var_hidden_size, num_layers=var_num_layers, output_size=var_output_size, dropout=var_dropout):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # usa a última saída da sequência
        return out

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def download_data(symbol, start="2015-01-01", end="2025-10-31"):
    """Baixa dados do Yahoo Finance e retorna DataFrame com as 5 colunas."""
    df = yf.download(symbol, start=start, end=end)
    if df is None or df.empty:
        raise ValueError(f"Sem dados para {symbol}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def create_sequences(data, look_back):
    """
    Cria janelas temporais com base nas 5 features.
    O modelo vai prever apenas o fechamento (índice 3 = Close).
    Input:
      data: np.array shape (n_samples, 5)
    Output:
      X: (n_windows, look_back, 5)
      y: (n_windows,)  <-- close normalizado
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])     # janela de look_back x 5
        y.append(data[i + look_back, 3])    # coluna 3 = Close
    return np.array(X), np.array(y)

# ============================================================
# FUNÇÃO PRINCIPAL DE TREINO
# ============================================================
def train(symbol, start, end, look_back=60, epochs=50, batch_size=32, lr=0.001, model_dir=None):
    # 1️Baixar os dados
    df = download_data(symbol, start, end)
    values = df.values.astype("float32")  # shape (n_days, 5)

    # 2️Normalização (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)  # shape (n_days, 5)

    # 3️Criar sequências
    X, y = create_sequences(scaled, look_back)  # X:(N,look_back,5), y:(N,)

    if len(X) == 0:
        raise ValueError("Não há dados suficientes para criar sequências com look_back=" + str(look_back))

    # Dividir em treino/validação/teste
    total = len(X)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    # Converter para tensores e garantir shapes (N, look_back, 5) e (N,1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)  # (N,1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    # keep y_test as numpy for metric inversion later

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    # Criar modelo
    model = StockLSTM(input_size=var_input_size, hidden_size=var_hidden_size, num_layers=var_num_layers, output_size=var_output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None

    log_path = Path(model_dir).parent / "training.log"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Iniciando treino: {symbol}\n")

    # Treinamento
    print(f"[TRAIN] Treinando {symbol} com 5 features...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)            # shape (batch,1)
            loss = criterion(preds, yb) # yb shape (batch,1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validação
        model.eval()
        with torch.no_grad():
            if X_val_t.shape[0] > 0:
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
            else:
                val_loss = float("nan")

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if not math.isnan(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Nenhum estado de modelo foi salvo (verifique os dados e a validação).")

    # Avaliação final (usar best_state)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).cpu().numpy()  # shape (n_test,1)

    # reconstruir matriz completa (n_test,5) para inverse_transform
    n_test = test_preds.shape[0]
    # preds -> colocar na coluna 'Close' (índice 3)
    preds_full = np.zeros((n_test, 5), dtype="float32")
    preds_full[:, 3] = test_preds.reshape(-1)

    # y_test (normalizado) -> colocar na coluna 'Close' também
    y_test_arr = np.array(y_test).reshape(-1, 1).astype("float32")  # shape (n_test,1)
    y_full = np.zeros((y_test_arr.shape[0], 5), dtype="float32")
    y_full[:, 3] = y_test_arr.reshape(-1)

    # Inverter escala
    preds_inv = scaler.inverse_transform(preds_full)[:, 3]
    y_true_inv = scaler.inverse_transform(y_full)[:, 3]

    mae = mean_absolute_error(y_true_inv, preds_inv)
    rmse = math.sqrt(mean_squared_error(y_true_inv, preds_inv))
    mape = np.mean(np.abs((y_true_inv - preds_inv) / y_true_inv)) * 100  

    print(f"[EVAL] MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.2f}%")

    # Adiciona log de finalização de treino
    
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Treino finalizado: {symbol} | MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%\n")

    # Salvar modelo e scaler
    if model_dir is None:
        model_dir = str(Path(__file__).resolve().parent / "ml_models")
    model_path = Path(model_dir) / symbol.replace("/", "_")
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path / "model.pt")
    joblib.dump(scaler, model_path / "scaler.pkl")

    print(f"[SAVE] Modelo salvo em {model_path.resolve()}")
    mae = float(mae)
    rmse = float(rmse)
    mape = float(mape)

    return {"symbol": symbol, "mae": mae, "rmse": rmse, "mape": mape}  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--look_back", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "ml_models"))
    args = parser.parse_args()

    train(args.symbol, args.start, args.end, args.look_back, args.epochs, args.batch_size, model_dir=args.model_dir)
