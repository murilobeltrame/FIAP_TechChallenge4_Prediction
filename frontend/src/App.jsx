import React, { useEffect, useState } from "react";
import MarketSelector from "./components/MarketSelector";
import TickerSelector from "./components/TickerSelector";
import QuoteResult from "./components/QuoteResult";
import Prediction from "./components/Prediction";

export default function App() {
  const [market, setMarket] = useState("");
  const [tickers, setTickers] = useState([]);
  const [selected, setSelected] = useState("");
  const [quote, setQuote] = useState(null);
  const [loadingTickers, setLoadingTickers] = useState(false);
  const [loadingQuote, setLoadingQuote] = useState(false);
  const [error, setError] = useState(null);
  const [trainedTickers, setTrainedTickers] = useState([]);
  const [loadingTrained, setLoadingTrained] = useState(false);

  // =========================================================
  // Carrega lista de tickers treinados no backend
  // =========================================================
  useEffect(() => {
    const fetchTrainedTickers = async () => {
      try {
        setLoadingTrained(true);
        const res = await fetch("http://127.0.0.1:8000/api/trained-tickers");
        if (!res.ok) throw new Error("Falha ao carregar tickers treinados");
        const data = await res.json();
        setTrainedTickers(data.trained_tickers || []);
      } catch (err) {
        console.error(err);
        setTrainedTickers([]);
      } finally {
        setLoadingTrained(false);
      }
    };
    fetchTrainedTickers();
  }, []);

  // =========================================================
  // Carrega lista de tickers do mercado selecionado
  // =========================================================
  useEffect(() => {
    if (!market) {
      setTickers([]);
      setSelected("");
      return;
    }

    const fetchTickers = async () => {
      try {
        setError(null);
        setLoadingTickers(true);
        const res = await fetch(`http://127.0.0.1:8000/api/stocks?market=${market}`);
        if (!res.ok) throw new Error("Falha ao carregar tickers");
        const data = await res.json();
        setTickers(data.tickers || []);
      } catch (err) {
        console.error(err);
        setError(err.message || "Erro desconhecido");
      } finally {
        setLoadingTickers(false);
      }
    };
    fetchTickers();
  }, [market]);

  // =========================================================
  // Busca cotação atual e histórico do ticker selecionado
  // =========================================================
  const handleSearch = async () => {
    if (!selected) return setError("Selecione uma ação antes de pesquisar");
    try {
      setError(null);
      setLoadingQuote(true);
      setQuote(null);
      let selected_fixed = market === "br" ? selected + ".SA" : selected;
      const res = await fetch("http://127.0.0.1:8000/api/quote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: selected_fixed }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Erro ao buscar cotação");
      }
      const data = await res.json();
      setQuote(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Erro desconhecido");
    } finally {
      setLoadingQuote(false);
    }
  };

  // =========================================================
  // Renderização
  // =========================================================
  return (
    <div className="app-shell">
      <aside className="panel">
        <div className="header">
          <h1>Consulta de Ações</h1>
          <p>Escolha o mercado e a ação para visualizar dados e gráfico (últimos 30 dias).</p>
        </div>

        <div className="form-row">
          <label className="form-label">Mercado</label>
          <MarketSelector value={market} onChange={(v) => setMarket(v)} className="select" />
        </div>

        <div className="form-row">
          <label className="form-label">Ação</label>
          <TickerSelector
            tickers={tickers}
            value={selected}
            onChange={(v) => setSelected(v)}
            disabled={loadingTickers}
            className="select"
          />
        </div>

        <div className="actions">
          <button
            className="button button-primary"
            onClick={handleSearch}
            disabled={loadingQuote || !selected}
          >
            {loadingQuote ? "Carregando..." : "Pesquisar"}
          </button>
          {loadingTickers && <div style={{ marginLeft: 8 }} className="loader" />}
        </div>

        {error && <div className="error" role="alert">{error}</div>}

        <div className="trained-section">
          <h3 style={{ marginTop: "1.5rem" }}>Modelos treinados disponíveis</h3>
          {loadingTrained ? (
            <p>Carregando lista...</p>
          ) : trainedTickers.length ? (
            <ul>
              {trainedTickers.map((t) => (
                <li key={t} style={{ fontSize: "0.9rem" }}>
                  ✅ {t}
                </li>
              ))}
            </ul>
          ) : (
            <p>Nenhum modelo treinado encontrado.</p>
          )}
        </div>
      </aside>

      <main className="result-card">
        <QuoteResult quote={quote} loading={loadingQuote} />
        <Prediction trainedTickers={trainedTickers || []} />
      </main>
    </div>
  );
}
