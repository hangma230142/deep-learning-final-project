# Task 5 — Model Deployment & SaaS

## Cấu trúc folder
```
task5/
├── best_lstm_vn.keras     ← trained model
├── scaler_vn.pkl          ← MinMaxScaler
├── vn_features.json       ← feature names
├── api.py                 ← FastAPI backend (Task 5.1)
├── app.py                 ← Streamlit frontend (Task 5.2)
└── requirements.txt
```

## Setup
```bash
cd task5
pip install -r requirements.txt
```

## Task 5.1 — Chạy API
```bash
uvicorn api:app --reload
```
API chạy tại: http://localhost:8000

### Test API
```bash
curl http://localhost:8000/health
```

## Task 5.2 — Chạy Web App
Mở terminal mới:
```bash
streamlit run app.py
```
Web chạy tại: http://localhost:8501

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API info |
| GET | /health | Health check |
| POST | /predict | Predict next day close price |

### Sample Request
```json
POST /predict
{
  "window": [[25000, 25500, 24500, 25200, 1000000], ...]  // 60 rows x 5 features
}
```

### Sample Response
```json
{
  "predicted_close": 25350.5,
  "unit": "VND"
}
```