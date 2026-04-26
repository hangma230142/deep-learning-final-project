import streamlit as st
import requests
import numpy as np
import pandas as pd
import json

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="VN Stock Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff9d;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00ff9d;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00ff9d, #00b4d8);
        color: #0a0e1a;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        width: 100%;
        cursor: pointer;
    }

    .info-box {
        background: #0f172a;
        border-left: 3px solid #00ff9d;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/predict"

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">📈 VN Stock Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning · LSTM · Vietnam Market</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL)
    st.markdown("---")
    st.markdown("### 📋 How to use")
    st.markdown("""
    1. Upload CSV hoặc nhập giá thủ công
    2. Cần **60 rows** dữ liệu lịch sử
    3. Bấm **Predict** để dự đoán
    4. Kết quả là giá Close ngày tiếp theo
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Model**: LSTM (units=128)
    - **Window**: 60 ngày
    - **Features**: Open/High/Low/Close/Volume
    - **Target**: Close (VND)
    """)

# ── Input method ──────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📂 Upload CSV")
    uploaded = st.file_uploader(
        "Upload file CSV (60 rows, 5 cột: Open/High/Low/Close/Volume)",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Đọc được {len(df)} rows, {len(df.columns)} cột")
        st.dataframe(df.tail(10), use_container_width=True)

with col2:
    st.markdown("### ✏️ Nhập giá thủ công (5 ngày gần nhất)")
    st.markdown('<div class="info-box">Nếu upload CSV, phần này sẽ bị bỏ qua.</div>', unsafe_allow_html=True)

    manual_data = []
    for i in range(5):
        cols = st.columns(5)
        row = [
            cols[0].number_input(f"Open {i+1}", value=25000.0, key=f"o{i}"),
            cols[1].number_input(f"High {i+1}", value=25500.0, key=f"h{i}"),
            cols[2].number_input(f"Low {i+1}",  value=24500.0, key=f"l{i}"),
            cols[3].number_input(f"Close {i+1}", value=25200.0, key=f"c{i}"),
            cols[4].number_input(f"Vol {i+1}",  value=1000000.0, key=f"v{i}"),
        ]
        manual_data.append(row)

# ── Predict button ────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🚀 Predict Next Day Close Price")

if predict_btn:
    try:
        # Prepare data
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            features = ["Open", "High", "Low", "Close", "Volume"]
            missing = [f for f in features if f not in df.columns]
            if missing:
                st.error(f"❌ CSV thiếu cột: {missing}")
                st.stop()
            if len(df) < 60:
                st.error(f"❌ Cần ít nhất 60 rows, hiện có {len(df)}")
                st.stop()
            window = df[features].tail(60).values.tolist()
        else:
            # Pad manual data to 60 rows by repeating
            arr = np.array(manual_data)
            arr = np.tile(arr, (12, 1))[:60]
            window = arr.tolist()

        # Call API
        with st.spinner("🔄 Đang dự đoán..."):
            resp = requests.post(api_url, json={"window": window}, timeout=30)

        if resp.status_code == 200:
            result = resp.json()
            predicted = result["predicted_close"]

            # Get last close for comparison
            if uploaded is not None:
                last_close = df["Close"].iloc[-1]
            else:
                last_close = manual_data[-1][3]

            change = predicted - last_close
            pct = (change / last_close) * 100

            st.markdown("---")
            st.markdown("### 🎯 Kết quả dự đoán")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Close</div>
                    <div class="metric-value">{predicted:,.0f}</div>
                    <div class="metric-label">VND</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                color = "#00ff9d" if change >= 0 else "#ff4d6d"
                arrow = "▲" if change >= 0 else "▼"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Change</div>
                    <div class="metric-value" style="color:{color}">{arrow} {abs(change):,.0f}</div>
                    <div class="metric-label">VND</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Change %</div>
                    <div class="metric-value" style="color:{color}">{pct:+.2f}%</div>
                    <div class="metric-label">vs Last Close</div>
                </div>""", unsafe_allow_html=True)

            # Chart
            if uploaded is not None:
                st.markdown("### 📊 Price History + Prediction")
                chart_df = df["Close"].tail(30).reset_index(drop=True)
                pred_point = pd.Series([None] * len(chart_df) + [predicted])
                hist_point = pd.concat([chart_df, pd.Series([None])], ignore_index=True)
                chart_data = pd.DataFrame({"Historical": hist_point, "Predicted": pred_point})
                st.line_chart(chart_data, use_container_width=True)

        else:
            st.error(f"❌ API Error {resp.status_code}: {resp.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Không kết nối được API. Hãy chạy: `uvicorn api:app --reload`")
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")