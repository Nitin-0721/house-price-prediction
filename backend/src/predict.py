import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def predict_price(data: dict) -> float:
    try:
        features = np.array([
            data["OverallQual"],
            data["GrLivArea"],
            data["GarageCars"],
            data["GarageArea"],
            data["TotalBsmtSF"],
            data["FirstFlrSF"],
            data["FullBath"],
            data["YearBuilt"],
            data["YrSold"],
            data["TotRmsAbvGrd"],
        ]).reshape(1, -1)

        features = scaler.transform(features)
        prediction = model.predict(features)
        price = float(prediction[0])

        # If price looks too small (model trained on thousands), multiply
        if price < 10000:
            price = price * 1000

        return round(price, 2)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")