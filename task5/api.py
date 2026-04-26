import numpy as np
import pickle
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import tensorflow as tf

app = FastAPI(title="Vietnam Stock Price Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler on startup
model = tf.keras.models.load_model("best_lstm_vn.keras")
with open("scaler_vn.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("vn_features.json") as f:
    VN_FEATURES = json.load(f)

CLOSE_IDX = VN_FEATURES.index("Close")
N_FEATURES = len(VN_FEATURES)  # 5


class PredictRequest(BaseModel):
    # 60 rows x 5 features: [[Open, High, Low, Close, Volume], ...]
    window: List[List[float]]


class PredictResponse(BaseModel):
    predicted_close: float
    unit: str = "VND"


def inverse_scale(scaler, value, feature_idx, n_features):
    dummy = np.zeros((1, n_features))
    dummy[0, feature_idx] = value
    return scaler.inverse_transform(dummy)[0, feature_idx]


@app.get("/")
def root():
    return {"message": "Vietnam Stock Predictor API", "features": VN_FEATURES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    window = np.array(req.window)
    if window.shape != (60, N_FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Input must be shape (60, {N_FEATURES}), got {window.shape}"
        )

    X = window.reshape(1, 60, N_FEATURES)
    pred_scaled = model.predict(X, verbose=0)[0][0]
    pred_price = inverse_scale(scaler, pred_scaled, CLOSE_IDX, N_FEATURES)

    return PredictResponse(predicted_close=round(float(pred_price), 2))


@app.get("/health")
def health():
    return {"status": "ok", "model": "best_lstm_vn", "window_size": 60}