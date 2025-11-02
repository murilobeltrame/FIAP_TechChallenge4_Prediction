from fastapi import APIRouter, HTTPException
from services.stock_provider import StockProvider
from schemas.stock import QuoteRequest, QuoteResponse

router = APIRouter()
provider = StockProvider()

@router.post("/quote", response_model=QuoteResponse)
async def get_quote(req: QuoteRequest):
    """
    Retorna os valores da ação pesquisada, referente aos ultimos 30 dias.
    """
    try:
        print(req.ticker)
        data = provider.get_quote(req.ticker, days=30)
        # print(data)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
