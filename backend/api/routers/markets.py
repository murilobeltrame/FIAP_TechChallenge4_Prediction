from fastapi import APIRouter, HTTPException, Query
from services.stock_provider import StockProvider
from schemas.stock import TickerList

router = APIRouter()
provider = StockProvider()

@router.get("/stocks", response_model=TickerList)
async def list_stocks(market: str = Query(..., description="br or us")):
    """
    Retorna uma lista com todos os tickers baseado no mercado selecionado.
    Os mercados atualmente disponíveis são BR e US.
    """
    try:
        items = provider.get_tickers_by_market(market)
        return {"market": market, "tickers": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
