import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

export default function QuoteResult({ quote, loading }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = null;
    }
    if (!quote || !quote.prices || quote.prices.length === 0) return;

    const ctx = canvasRef.current.getContext("2d");
    const labels = quote.prices.map(p => p.date);
    const data = quote.prices.map(p => p.close);

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: `${quote.ticker} - Preço (close)`,
          data,
          tension: 0.25,
          borderWidth: 2,
          // pointRadius: 0,
          fill: true,
          backgroundColor: "rgba(37,99,235,0.12)",
          borderColor: "rgba(37,99,235,1)"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [quote]);

  if (loading) {
    return (
      <div style={{display:"flex",alignItems:"center",justifyContent:"center",height:"100%"}}>
        <div style={{textAlign:"center"}}>
          <div className="loader" />
          <div style={{marginTop:8,color:"#6b7280"}}>Buscando dados...</div>
        </div>
      </div>
    );
  }

  if (!quote) {
    return <div className="empty-state">Selecione um mercado e uma ação para visualizar dados.</div>;
  }

  return (
    <div>
      <div className="result-header">
        <div className="stock-title">
          <h2>{quote.ticker}</h2>
          <span>{quote.name || "—"}</span>
        </div>
        <div className="meta">
          <div>{quote.sector || "Setor: —"}</div>
          <div>{quote.industry || "Indústria: —"}</div>
        </div>
      </div>

      <div className="chart-wrapper" style={{height: 320}}>
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
