import streamlit as st
import requests
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VN Stock Price Predictor",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #001D3F;
    color: #ffffff;
}

/* Header */
.page-header {
    padding: 2.5rem 0 1.5rem 0;
    color: #ffffff;
    border-bottom: 2px solid #1a1a2e;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #fffff;
    letter-spacing: -0.5px;
    margin: 0;
}
.page-desc {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #fffff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e5e7eb;
}

/* Info box */
.info-box {
    background: #eef2ff;
    border-left: 3px solid #4f46e5;
    padding: 0.85rem 1.2rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #3730a3;
    margin: 0.75rem 0 1.25rem 0;
    line-height: 1.6;
}

/* Result cards */
.result-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
}
.result-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.result-card-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}
.result-card-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1.1;
}
.result-card-unit {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 0.3rem;
}
.positive { color: #059669; }
.negative { color: #dc2626; }

/* Predict button */
.stButton > button {
    background-color: #1a1a2e;
    color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    border: none;
    border-radius: 6px;
    padding: 0.7rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #2d2d4e;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #001D3F;
    border-right: 1px solid #e5e7eb;
}
.sidebar-section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.6rem;
}
.sidebar-item {
    font-size: 0.85rem;
    color: #ffffff;
    padding: 0.25rem 0;
    line-height: 1.6;
}
.model-tag {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #374151;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.15rem 0.15rem 0;
}

/* Divider */
hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
DEFAULT_API = "http://localhost:8000/predict"
FEATURES    = ["Open", "High", "Low", "Close", "Volume"]
WINDOW_SIZE = 60

# ── Session state ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Configuration</div>', unsafe_allow_html=True)
    api_url = st.text_input("API Endpoint", value=DEFAULT_API, label_visibility="collapsed")
    st.markdown(f'<div class="sidebar-item" style="color:#6b7280;font-size:0.78rem;">{api_url}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">How It Works</div>', unsafe_allow_html=True)
    for step in [
        "Upload a CSV file with at least 60 rows",
        "Columns required: Open, High, Low, Close, Volume",
        "The model uses the most recent 60 rows as input",
        "Click Predict to get the next-day Close price",
    ]:
        st.markdown(f'<div class="sidebar-item">— {step}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Model Details</div>', unsafe_allow_html=True)
    for tag in ["GRU · units=128", "Window: 60 days", "Target: Close (VND)", "Trained on HPG"]:
        st.markdown(f'<span class="model-tag">{tag}</span>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">VN Stock Price Predictor</div>
    <div class="page-desc">
        Deep learning model for Vietnam stock market · Next-day Close price prediction
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input columns ─────────────────────────────────────────────
col_upload, col_manual = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-label">Upload Historical Data</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "CSV file — at least 60 rows, columns: Open / High / Low / Close / Volume",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            elif len(df) < WINDOW_SIZE:
                st.error(f"Need at least {WINDOW_SIZE} rows. File has {len(df)}.")
            else:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows successfully.")
                st.dataframe(
                    df[FEATURES].tail(10).style.format("{:,.0f}"),
                    use_container_width=True,
                    height=280,
                )
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"Using previously uploaded file — {len(df)} rows.")
        st.dataframe(
            df[FEATURES].tail(10).style.format("{:,.0f}"),
            use_container_width=True,
            height=280,
        )

with col_manual:
    st.markdown('<div class="section-label">Manual Input (Last 5 Days)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        If a CSV file is uploaded above, manual input is ignored.
        Use this option only when no CSV is available.
        The 5 rows will be repeated to fill the 60-day window.
    </div>
    """, unsafe_allow_html=True)

    header = st.columns(5)
    for col, name in zip(header, ["Open", "High", "Low", "Close", "Volume"]):
        col.markdown(f"<div style='font-size:0.75rem;font-weight:600;color:#6b7280;text-align:center'>{name}</div>", unsafe_allow_html=True)

    manual_rows = []
    defaults = {"Open": 25000.0, "High": 25500.0, "Low": 24500.0, "Close": 25200.0, "Volume": 1000000.0}
    for i in range(5):
        cols = st.columns(5)
        row = [
            cols[0].number_input("", value=defaults["Open"],   key=f"o{i}", label_visibility="collapsed"),
            cols[1].number_input("", value=defaults["High"],   key=f"h{i}", label_visibility="collapsed"),
            cols[2].number_input("", value=defaults["Low"],    key=f"l{i}", label_visibility="collapsed"),
            cols[3].number_input("", value=defaults["Close"],  key=f"c{i}", label_visibility="collapsed"),
            cols[4].number_input("", value=defaults["Volume"], key=f"v{i}", label_visibility="collapsed"),
        ]
        manual_rows.append(row)

# ── Predict ───────────────────────────────────────────────────
st.markdown("""
    <style>
    div.stButton { text-align: center; }
    div.stButton > button { margin: auto; display: block; }
    </style>
""", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("RUN PREDICTION", use_container_width=True)

if predict:
    # Build window
    if st.session_state.df is not None:
        df = st.session_state.df
        window = df[FEATURES].tail(WINDOW_SIZE).values.tolist()
        last_close = float(df["Close"].iloc[-1])
    else:
        arr = np.tile(np.array(manual_rows), (12, 1))[:WINDOW_SIZE]
        window = arr.tolist()
        last_close = float(manual_rows[-1][3])

    with st.spinner("Calling prediction API..."):
        try:
            resp = requests.post(api_url, json={"window": window}, timeout=30)

            if resp.status_code == 200:
                predicted = resp.json()["predicted_close"]
                change    = predicted - last_close
                pct       = (change / last_close) * 100 if last_close != 0 else 0
                direction = "positive" if change >= 0 else "negative"
                sign      = "+" if change >= 0 else ""

                st.markdown(f"""
                <div class="result-row">
                    <div class="result-card">
                        <div class="result-card-label">Predicted Close</div>
                        <div class="result-card-value">{predicted:,.0f}</div>
                        <div class="result-card-unit">VND — next trading day</div>
                    </div>
                    <div class="result-card">
                        <div class="result-card-label">Change vs Last Close</div>
                        <div class="result-card-value {direction}">{sign}{change:,.0f}</div>
                        <div class="result-card-unit">VND absolute</div>
                    </div>
                    <div class="result-card">
                        <div class="result-card-label">Percentage Change</div>
                        <div class="result-card-value {direction}">{sign}{pct:.2f}%</div>
                        <div class="result-card-unit">vs last close {last_close:,.0f} VND</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Chart
                if st.session_state.df is not None:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Price History + Prediction</div>', unsafe_allow_html=True)

                    history = st.session_state.df["Close"].tail(30).reset_index(drop=True)
                    chart_df = pd.DataFrame({
                        "Historical Close": pd.concat(
                            [history, pd.Series([None])], ignore_index=True
                        ),
                        "Predicted": pd.Series(
                            [None] * len(history) + [predicted]
                        ),
                    })
                    st.line_chart(chart_df, use_container_width=True, height=280)

            else:
                st.error(f"API returned error {resp.status_code}: {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the server is running: uvicorn api:app --reload")
        except Exception as e:
            st.error(f"Unexpected error: {e}")