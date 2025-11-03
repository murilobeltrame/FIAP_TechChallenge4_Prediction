# backend/services/lstm_service.py
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from pathlib import Path
import joblib

var_input_size = 5
var_hidden_size = 128
var_num_layers = 2
var_output_size = 1
var_dropout = 0.2

# ============================================================
# MODELO LSTM — deve ser idêntico ao do treino
# ============================================================
class StockLSTM(nn.Module):
    def __init__(self, input_size=var_input_size, hidden_size=var_hidden_size, num_layers=var_num_layers, output_size=var_output_size, dropout=var_dropout):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # usa o último passo da sequência
        return out

# ============================================================
# CAMINHO BASE DOS MODELOS SALVOS
# ============================================================
BASE_MODELS_DIR = Path(__file__).resolve().parents[1] / "ml_models"

# ============================================================
# FUNÇÃO DE CARREGAMENTO DO MODELO E SCALER
# ============================================================
def load_model_and_scaler(symbol):
    """Carrega o modelo e o scaler do diretório correspondente."""
    model_dir = BASE_MODELS_DIR / symbol.replace("/", "_")
    model_path = model_dir / "model.pt"
    scaler_path = model_dir / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Modelo ou scaler não encontrado para {symbol}")

    # Carrega modelo
    model = StockLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Carrega scaler
    scaler = joblib.load(scaler_path)
    return model, scaler

# ============================================================
# FUNÇÃO DE PREVISÃO FUTURA
# ============================================================
def predict_future(symbol: str, days: int = 7, look_back: int = 60):
    """
    Gera previsões futuras com base no último período de dados reais.
    Atualiza recursivamente a sequência usando o valor previsto de fechamento (Close).
    """
    model, scaler = load_model_and_scaler(symbol)

    # 🔽 Download dos dados mais recentes
    df = yf.download(symbol, period="2y")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    if len(df) < look_back:
        raise ValueError("Dados insuficientes para predição")

    # 🔢 Normaliza os dados
    data = df.values.astype("float32")
    scaled = scaler.transform(data)
    seq = scaled[-look_back:].tolist()  # últimos N dias como janela inicial

    preds = []
    for step in range(days):
        X = torch.tensor(seq[-look_back:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred_scaled = model(X).item()

        # ✅ Atualiza a sequência: substitui o fechamento (Close)
        new_row = seq[-1].copy()
        new_row[3] = y_pred_scaled

        # adiciona pequenas variações nas outras colunas
        new_row[0] *= np.random.uniform(0.998, 1.002)  # Open
        new_row[1] *= np.random.uniform(0.998, 1.002)  # High
        new_row[2] *= np.random.uniform(0.998, 1.002)  # Low
        new_row[4] *= np.random.uniform(0.95, 1.05)    # Volume

        # adiciona a nova linha à sequência
        seq.append(new_row)

        # 🔁 Reverte a escala apenas para o fechamento
        inv_full = np.zeros((1, 5))
        inv_full[0, 3] = y_pred_scaled
        inv = scaler.inverse_transform(inv_full)[0][3]

        preds.append(inv)

        print(f"[STEP {step+1}] Predição: {inv:.4f}")

    return preds
