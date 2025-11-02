import React, { useState, useRef } from "react";
import Chart from "chart.js/auto";

export default function Prediction({ trainedTickers = [] }) {
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(7);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const chartRef = useRef(null);
  const canvasRef = useRef(null);

  const handlePredict = async () => {
    if (!ticker) return alert("Escolha um ticker!");
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, days }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Erro na previsão");

      setPredictions(data.predictions);

      // --- render chart ---
      const labels = data.predictions.map((p) => p.date);
      const values = data.predictions.map((p) => Number(p.predicted).toFixed(2));

      const ctx = canvasRef.current.getContext("2d");
      if (chartRef.current) chartRef.current.destroy();

      chartRef.current = new Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: `${ticker} - Previsão (Próximos ${days} dias)`,
              data: values,
              borderWidth: 2,
              borderColor: "rgba(37,99,235,1)",
              backgroundColor: "rgba(0,123,255,0.1)",
              fill: true,
              tension: 0.3,
            },
          ],
        },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: "Data" } },
            y: { title: { display: true, text: "Preço Previsto" } },
          },
        },
      });
    } catch (e) {
      alert("Erro: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 20 }}>
      <h3>📈 Previsão LSTM (PyTorch)</h3>

      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          style={{ padding: "4px 8px", 
            color:"black",
            borderRadius: "8px",
            border: "1px solid rgba(15,23,42,0.07)",
            background: "#fff", 
          }}
        >
          <option value="">Selecione um modelo treinado</option>
          {trainedTickers.map((t, i) => (
            <option key={i} value={t}>
              {t}
            </option>
          ))}
        </select>

        <input
          type="number"
          min={1}
          max={30}
          value={days}
          onChange={(e) => setDays(parseInt(e.target.value))}
          style={{ width: "70px", 
            padding: "4px",
            color:"black",
            borderRadius: "8px",
            border: "1px solid rgba(15,23,42,0.07)",
            background: "#fff", 
          }}
        />

        <button
          onClick={handlePredict}
          disabled={loading || !ticker}
          style={{
            background: "linear-gradient(180deg, var(--primary), var(--primary-600))",
            color: "white",
            border: "none",
            padding: "6px 12px",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          {loading ? "Prevendo..." : "Prever"}
        </button>
      </div>

      <div style={{ height: 300, marginTop: 16 }}>
        <canvas ref={canvasRef}></canvas>
      </div>

      {predictions.length > 0 && (
        <p style={{ marginTop: 8, fontSize: "0.9rem" }}>
          Mostrando previsões para <strong>{ticker}</strong> (próximos{" "}
          {days} dias)
        </p>
      )}
    </div>
  );
}
