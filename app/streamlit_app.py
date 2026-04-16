import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIG

st.set_page_config(
    page_title="SignalGuard AI",
    layout="wide"
)

#  CSS

st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
}

/* Glow Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #38bdf8;
    text-shadow: 0 0 20px #38bdf8;
}

/* Animated Alert */
@keyframes pulse {
    0% {box-shadow: 0 0 10px red;}
    50% {box-shadow: 0 0 25px red;}
    100% {box-shadow: 0 0 10px red;}
}

.alert {
    background: rgba(255, 0, 0, 0.2);
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    animation: pulse 2s infinite;
}

/* Metric Glow */
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 15px rgba(56,189,248,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Scroll */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/final_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def load_metrics():
    with open("outputs/metrics/results.json") as f:
        return json.load(f)

df = load_data()
metrics = load_metrics()

# SIDEBAR

st.sidebar.title("⚙️ Controls")

sensor = st.sidebar.selectbox(
    "Sensor",
    ['temperature', 'vibration', 'pressure', 'rotation_speed', 'power_consumption']
)

model_map = {
    "Isolation Forest": "if_anomaly",
    "Z-score": "zscore_anomaly",
    "LSTM Autoencoder": "lstm_anomaly"
}

model_name = st.sidebar.selectbox("Model", list(model_map.keys()))
model_choice = model_map[model_name]

# HEADER

st.markdown('<div class="title">🚀 SignalGuard AI Dashboard</div>', unsafe_allow_html=True)

total_anomalies = int(df[model_choice].sum())

st.markdown(f"""
<div class="alert">
⚠️ {total_anomalies} Anomalies Detected!
</div>
""", unsafe_allow_html=True)


# METRICS

st.markdown("### 📊 Metrics")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown('<div class="metric-card">Accuracy<br><b>{:.2f}</b></div>'.format(metrics['isolation_forest']['accuracy']), unsafe_allow_html=True)

with m2:
    st.markdown('<div class="metric-card">Precision<br><b>{:.2f}</b></div>'.format(metrics['isolation_forest']['precision']), unsafe_allow_html=True)

with m3:
    st.markdown('<div class="metric-card">Recall<br><b>{:.2f}</b></div>'.format(metrics['isolation_forest']['recall']), unsafe_allow_html=True)

with m4:
    st.markdown('<div class="metric-card">F1 Score<br><b>{:.2f}</b></div>'.format(metrics['isolation_forest']['f1_score']), unsafe_allow_html=True)


# MAIN CHART

st.markdown("### 📈 Anomaly Detection")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df[sensor],
    mode='lines',
    line=dict(color='#38bdf8', width=2),
    name="Signal"
))

anomalies = df[df[model_choice] == 1]

fig.add_trace(go.Scatter(
    x=anomalies['timestamp'],
    y=anomalies[sensor],
    mode='markers',
    marker=dict(color='red', size=8),
    name="Anomaly"
))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=450
)

st.plotly_chart(fig, use_container_width=True)


# LOWER GRID

col1, col2, col3 = st.columns(3)

# ---- Stats ----
with col1:
    st.markdown("### 🚨 Stats")
    st.write(f"Total Anomalies: {total_anomalies}")
    st.write(f"Records: {len(df)}")

# ---- Alerts ----
with col2:
    st.markdown("### 🔔 Alerts")
    recent = anomalies.tail(5)

    for _, row in recent.iterrows():
        st.write(f"{row['timestamp']} → anomaly")

# ---- System ----
with col3:
    st.markdown("### ⚙️ System")
    st.write("🟢 Operational")
    st.write("CPU: 23%")
    st.write("Memory: 58%")


# PERFORMANCE

st.markdown("### 📊 Model Comparison")

perf_df = pd.DataFrame({
    "Model": ["IF", "Z-score", "LSTM"],
    "F1": [
        metrics['isolation_forest']['f1_score'],
        metrics['zscore']['f1_score'],
        metrics['lstm']['f1_score']
    ]
})

fig2 = px.bar(
    perf_df,
    x="Model",
    y="F1",
    color="Model",
    text_auto=True
)

fig2.update_layout(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig2, use_container_width=True)


# DATA

st.markdown("### 📂 Data Preview")
st.dataframe(df.head(100))

# FOOTER

st.markdown("---")
st.markdown("<center>Built by Deepika| SignalGuard AI</center>", unsafe_allow_html=True)