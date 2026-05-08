"""
dags/stock_prediction_pipeline.py  —  Task 5.3: Airflow DAG
Runs daily at 6:00 AM ICT (23:00 UTC previous day) on weekdays,
ensuring fresh predictions before Vietnam market opens at 9:00 AM.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.time_delta import TimeDeltaSensor

DEFAULT_ARGS = {
    "owner": "dl4ai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def run_lstm_predictions():
    """
    Load saved GRU model, fetch latest 60-day window from MongoDB,
    run inference, and write predictions back to MongoDB.
    """
    import numpy as np
    import pickle
    import tensorflow as tf
    from pymongo import MongoClient
    from datetime import datetime

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["stocks"]

    # Load model and scaler
    model = tf.keras.models.load_model("./task5/best_lstm_vn.keras")
    with open("./task5/scaler_vn.pkl", "rb") as f:
        scaler = pickle.load(f)

    N_FEATURES = 5   # Open, High, Low, Close, Volume
    TARGET_IDX = 3   # Close
    WINDOW_SIZE = 60

    # Fetch all tickers with enough data
    tickers = db["features"].distinct("ticker")

    predictions_written = 0
    for ticker in tickers:
        rows = list(
            db["features"]
            .find({"ticker": ticker})
            .sort("date", -1)
            .limit(WINDOW_SIZE)
        )
        if len(rows) < WINDOW_SIZE:
            continue

        # Chronological order
        rows = rows[::-1]
        window = np.array(
            [[r["open"], r["high"], r["low"], r["close"], r["volume"]] for r in rows],
            dtype=np.float32,
        )

        # Scale and predict
        window_scaled = scaler.transform(window)[np.newaxis]  # (1, 60, 5)
        pred_scaled = model.predict(window_scaled, verbose=0)[0, 0]

        # Inverse transform
        dummy = np.zeros((1, N_FEATURES))
        dummy[0, TARGET_IDX] = pred_scaled
        predicted_close = float(scaler.inverse_transform(dummy)[0, TARGET_IDX])

        # Write to predictions collection
        db["predictions"].update_one(
            {"ticker": ticker, "predicted_date": datetime.utcnow().date().isoformat()},
            {
                "$set": {
                    "ticker": ticker,
                    "predicted_close": predicted_close,
                    "predicted_date": datetime.utcnow().date().isoformat(),
                    "created_at": datetime.utcnow(),
                }
            },
            upsert=True,
        )
        predictions_written += 1

    client.close()
    print(f"Predictions written for {predictions_written} tickers.")


# ── DAG Definition ────────────────────────────────────────────
with DAG(
    dag_id="stock_prediction_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily Vietnam stock prediction pipeline",
    # Run at 23:00 UTC = 06:00 ICT, weekdays only
    schedule_interval="0 23 * * 0-4",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["dl4ai", "stock", "vietnam"],
) as dag:

    ingest_task = BashOperator(
        task_id="trigger_airbyte_sync",
        bash_command="""
            curl -s -X POST http://localhost:8001/api/v1/connections/sync \
                 -H "Content-Type: application/json" \
                 -d '{"connectionId": "YOUR_AIRBYTE_CONNECTION_ID"}'
        """,
    )

    wait_task = BashOperator(
        task_id="wait_for_airbyte",
        bash_command="sleep 120",  # Wait 2 minutes for sync to finish
    )

    dbt_run_task = BashOperator(
        task_id="run_dbt_transformations",
        bash_command="cd /opt/dbt/stock_project && dbt run --select stock_features",
    )

    predict_task = PythonOperator(
        task_id="run_predictions",
        python_callable=run_lstm_predictions,
    )

    notify_task = BashOperator(
        task_id="log_completion",
        bash_command='echo "Pipeline complete at $(date). Predictions ready."',
    )

    ingest_task >> wait_task >> dbt_run_task >> predict_task >> notify_task