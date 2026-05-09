from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import PredictRequest, PredictResponse
from src.predict import predict_price

app = FastAPI(title="House Price API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "House Price API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/api/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        price = predict_price(data.dict())
        return PredictResponse(
            predicted_price=price,
            confidence_low=round(price * 0.92, 2),
            confidence_high=round(price * 1.08, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))