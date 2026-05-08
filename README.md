# Deep Learning for AI — Final Project
## Time-series Data and Application to Stock Markets

**CS313 Deep Learning for Artificial Intelligence | SPRING 2026**  
**Student:** Hang Ma

---

## Project Overview

This project builds a complete deep learning pipeline for stock market analysis using LSTM, GRU, and CNN architectures. It covers five main tasks:

| Task | Description | Score |
|------|-------------|-------|
| Task 1 | Nasdaq stock price prediction (AAPL) | 15% |
| Task 2 | Vietnam stock price prediction (HPG, FPT, VNM, VIC, VCB) | 15% |
| Task 3 | Trading signal identification (Buy/Sell) for Vietnam market | 20% |
| Task 4 | Portfolio composition and risk management | 30% |
| Task 5 | Model deployment, SaaS web app, AI automation workflow | 30% (extra) |

---

## Repository Structure

```
deep-learning-final-project/
├── notebook/
│   └── Final_project_DL4AI.ipynb     # Main notebook: Task 1 to Task 4
├── task5/
│   ├── api.py                         # Task 5.1: FastAPI REST API
│   ├── app.py                         # Task 5.2: Streamlit web app
│   ├── best_lstm_vn.keras             # Trained GRU model (saved)
│   ├── scaler_vn.pkl                  # MinMaxScaler fitted on training data
│   ├── vn_features.json               # Feature list metadata
│   └── requirements.txt               # Python dependencies for Task 5
├── dags/
│   └── stock_prediction_pipeline.py  # Task 5.3: Airflow DAG
├── dbt_models/
│   └── stock_features.sql            # Task 5.3: dbt feature transformation
└── README.md
```

---

## Setup and Installation

### Requirements

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/deep-learning-final-project.git
cd deep-learning-final-project
```

### 2. Run the notebook (Task 1–4)

Open `notebook/Final_project_DL4AI.ipynb` in Google Colab or Jupyter:

```bash
# Option A: Jupyter locally
pip install jupyter tensorflow scikit-learn pandas numpy matplotlib imbalanced-learn tqdm
jupyter notebook notebook/Final_project_DL4AI.ipynb

# Option B: Upload to Google Colab (recommended)
# Go to https://colab.research.google.com → Upload → select the .ipynb file
```

> **Note:** The notebook expects the Vietnam and Nasdaq datasets to be mounted at the paths configured in the first cell. Update the `DATA_DIR` variables if needed.

### 3. Run Task 5 (Model Deployment + Web App)

```bash
cd task5
pip install -r requirements.txt
```

**Start the API server (Terminal 1):**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Start the web app (Terminal 2):**
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## Task Descriptions

### Task 1 — Nasdaq Stock Price Prediction
- **1.1** Multi-feature LSTM (6 features: Open, High, Low, Close, Adj Close, Volume)
- **1.2** N-th day ahead forecast (N = 3, 7)
- **1.3** K consecutive days forecast (K = 3, 7)
- Hyperparameter tuning via grid search (units ∈ {32, 64, 128}, lr ∈ {1e-3, 5e-4})
- Best result: MAE = 3.39 USD, RMSE = 5.37 USD (AAPL)

### Task 2 — Vietnam Stock Price Prediction
- Same pipeline as Task 1 applied to Vietnam data (HPG as primary stock)
- Architecture comparison: LSTM vs GRU vs CNN across 5 representative companies
- Best result: GRU achieves MAE = 810 VND on HPG (42% better than LSTM)

### Task 3 — Trading Signal Identification
- Buy and Sell signal classification using LSTM, GRU, CNN, CNN-LSTM
- SMOTE oversampling to handle class imbalance (1,989 samples per class)
- Lookforward labeling (local min/max within ±5-day window)
- Best result: CNN-LSTM achieves F1 = 0.4780 for Sell signal

### Task 4 — Portfolio Composition and Risk Management
- **4.1** Profitable stock selection via backtesting (50 Vietnam companies)
- **4.2** Risk scoring using 6 metrics (volatility, drawdown, VaR, Sharpe, etc.)
- **4.3** Two portfolio strategies: Aggressive (Sharpe 0.48) vs Prudent (Sharpe 0.59)

### Task 5 — Deployment and AI Automation
- **5.1** FastAPI REST API serving the trained GRU model at `POST /predict`
- **5.2** Streamlit web interface with CSV upload and real-time prediction
- **5.3** Airflow DAG + dbt + MongoDB automated daily pipeline

---

## API Usage (Task 5.1)

**Endpoint:** `POST http://localhost:8000/predict`

**Request body:**
```json
{
  "window": [
    [25000, 25500, 24500, 25200, 1000000],
    ...
  ]
}
```
*(60 rows of [Open, High, Low, Close, Volume])*

**Response:**
```json
{
  "predicted_close": 25387.50,
  "unit": "VND"
}
```

**Health check:** `GET http://localhost:8000/health`

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow / Keras |
| Machine Learning | scikit-learn |
| Data Processing | pandas, numpy |
| Class Imbalance | imbalanced-learn (SMOTE) |
| API Server | FastAPI, Uvicorn |
| Web Interface | Streamlit |
| Pipeline Orchestration | Apache Airflow |
| Data Ingestion | Airbyte |
| Data Transformation | dbt |
| Database | MongoDB |

---

## Key Results Summary

| Task | Metric | Value |
|------|--------|-------|
| Task 1 — AAPL next-day | MAE | 3.39 USD |
| Task 2 — HPG next-day (GRU) | MAE | 810 VND |
| Task 3 — Buy signal (LSTM) | F1 | 0.4051 |
| Task 3 — Sell signal (CNN-LSTM) | F1 | 0.4780 |
| Task 4 — HPG strategy vs B&H | Return | +49.4% vs −4.1% |
| Task 4 — Prudent portfolio | Sharpe | 0.59 |

---

## References

1. Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997
2. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder", arXiv:1406.1078, 2014
3. López de Prado, "Advances in Financial Machine Learning", Wiley, 2018
4. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", JAIR, 2002