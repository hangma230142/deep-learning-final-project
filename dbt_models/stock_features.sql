-- dbt_models/stock_features.sql
-- Task 5.3: dbt transformation model
-- Transforms raw OHLCV data into feature table with technical indicators.
-- Run with: dbt run --select stock_features

{{ config(materialized='table', schema='analytics') }}

WITH base AS (
    SELECT
        ticker,
        date,
        open,
        high,
        low,
        close,
        volume
    FROM {{ source('raw', 'stock_raw') }}
    WHERE close > 0
      AND volume > 0
),

with_indicators AS (
    SELECT
        ticker,
        date,
        open,
        high,
        low,
        close,
        volume,

        -- Simple Moving Average 5-day
        AVG(close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS sma_5,

        -- Simple Moving Average 20-day
        AVG(close) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS sma_20,

        -- Daily return
        (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
            / NULLIF(LAG(close) OVER (PARTITION BY ticker ORDER BY date), 0)
            AS daily_return,

        -- Rolling 20-day volatility (std of daily returns)
        STDDEV(
            (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
            / NULLIF(LAG(close) OVER (PARTITION BY ticker ORDER BY date), 0)
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS volatility_20d,

        -- Average gain for RSI (14-day)
        AVG(
            CASE
                WHEN close - LAG(close) OVER (PARTITION BY ticker ORDER BY date) > 0
                THEN close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)
                ELSE 0
            END
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_gain_14,

        -- Average loss for RSI (14-day)
        AVG(
            CASE
                WHEN close - LAG(close) OVER (PARTITION BY ticker ORDER BY date) < 0
                THEN ABS(close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
                ELSE 0
            END
        ) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) AS avg_loss_14

    FROM base
),

with_rsi AS (
    SELECT
        *,
        -- RSI (14-day)
        CASE
            WHEN avg_loss_14 = 0 THEN 100
            ELSE 100 - (100 / (1 + avg_gain_14 / NULLIF(avg_loss_14, 0)))
        END AS rsi_14

    FROM with_indicators
)

SELECT
    ticker,
    date,
    open,
    high,
    low,
    close,
    volume,
    sma_5,
    sma_20,
    rsi_14,
    daily_return,
    volatility_20d,
    -- Golden Cross flag (SMA5 crossed above SMA20)
    CASE
        WHEN sma_5 > sma_20
         AND LAG(sma_5) OVER (PARTITION BY ticker ORDER BY date)
             <= LAG(sma_20) OVER (PARTITION BY ticker ORDER BY date)
        THEN 1 ELSE 0
    END AS golden_cross,
    -- Death Cross flag (SMA5 crossed below SMA20)
    CASE
        WHEN sma_5 < sma_20
         AND LAG(sma_5) OVER (PARTITION BY ticker ORDER BY date)
             >= LAG(sma_20) OVER (PARTITION BY ticker ORDER BY date)
        THEN 1 ELSE 0
    END AS death_cross

FROM with_rsi
WHERE sma_20 IS NOT NULL  -- Drop warm-up rows
ORDER BY ticker, date