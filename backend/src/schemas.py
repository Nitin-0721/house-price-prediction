from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    OverallQual: int = Field(..., ge=1, le=10)
    GrLivArea: float = Field(..., gt=0)
    GarageCars: int = Field(..., ge=0, le=5)
    GarageArea: float = Field(..., ge=0)
    TotalBsmtSF: float = Field(..., ge=0)
    FirstFlrSF: float = Field(..., ge=0)
    FullBath: int = Field(..., ge=0, le=5)
    YearBuilt: int = Field(..., ge=1800, le=2024)
    YrSold: int = Field(..., ge=2000, le=2024)
    TotRmsAbvGrd: int = Field(..., ge=1, le=20)

class PredictResponse(BaseModel):
    predicted_price: float
    confidence_low: float
    confidence_high: float
    currency: str = "USD"