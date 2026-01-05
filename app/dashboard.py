import streamlit as st
import pandas as pd
import time
import plotly.express as px
import numpy as np

# Page Config
st.set_page_config(
    page_title="AjayDataLabs | Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Real-Time Financial Fraud Detection")
st.markdown("### 🔴 Live System Monitor (Cloud Demo)")

# --- 1. SESSION STATE SETUP ---
# This acts as our "Temporary Database" in the cloud memory
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['timestamp', 'amount', 'anomaly_score', 'prediction'])

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("🔧 Control Panel")
run_simulation = st.sidebar.checkbox("Run Live Simulation", value=True)
speed = st.sidebar.slider("Refresh Speed (seconds)", 0.1, 2.0, 1.0)

# --- 3. SIMULATION LOGIC (No DB Required) ---
def generate_fake_data():
    """Generates a single transaction for the demo"""
    now = time.strftime("%H:%M:%S")
    
    # 95% Normal, 5% Fraud
    if np.random.random() > 0.95: 
        amount = np.random.normal(500, 100) # High amount
        score = np.random.uniform(-0.3, -0.01) # Negative score = Anomaly
        pred = "Anomaly"
    else:
        amount = np.random.normal(50, 20)   # Normal amount
        score = np.random.uniform(0.0, 0.3)  # Positive score = Normal
        pred = "Normal"
        
    return {"timestamp": now, "amount": abs(amount), "anomaly_score": score, "prediction": pred}

# --- 4. DASHBOARD LOOP ---
placeholder = st.empty()

if run_simulation:
    # Generate 1 new row of data
    new_row = generate_fake_data()
    
    # Add to our "memory" dataframe (Keep last 200 rows only)
    st.session_state.data = pd.concat([pd.DataFrame([new_row]), st.session_state.data], ignore_index=True).head(200)

    # UI LAYOUT
    df = st.session_state.data
    anomalies = df[df['prediction'] == 'Anomaly']
    
    with placeholder.container():
        # KPI Row
        k1, k2, k3 = st.columns(3)
        k1.metric("Transactions Processed", len(df))
        k2.metric("🚨 Anomalies Detected", len(anomalies), delta_color="inverse")
        k3.metric("System Status", "ACTIVE" if len(anomalies) < 5 else "HIGH ALERT")

        # Charts
        c1, c2 = st.columns(2)
        
        # Chart 1: Anomaly Scores
        fig_line = px.line(df, x='timestamp', y='anomaly_score', title="Live Anomaly Score Stream", markers=True)
        fig_line.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Threshold")
        c1.plotly_chart(fig_line, use_container_width=True)
        
        # Chart 2: Amount vs Risk
        fig_scat = px.scatter(df, x='amount', y='anomaly_score', color='prediction',
                              color_discrete_map={"Normal": "blue", "Anomaly": "red"},
                              title="Transaction Amount vs. Risk Score")
        c2.plotly_chart(fig_scat, use_container_width=True)

        # Table
        st.dataframe(df.style.applymap(lambda x: 'color: red; font-weight: bold' if x == 'Anomaly' else 'color: green', subset=['prediction']), use_container_width=True)

    # Wait before next update
    time.sleep(speed)
    st.rerun()