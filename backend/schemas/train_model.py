from pydantic import BaseModel

class TrainRequest(BaseModel):
    symbol: str
    start: str = "2015-01-01"
    end: str = "2025-10-31"
    look_back: int = 60
    epochs: int = 50
    batch_size: int = 32