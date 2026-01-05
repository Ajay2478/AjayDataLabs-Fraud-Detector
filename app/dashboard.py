import streamlit as st
import pandas as pd
import plotly.express as px
import time
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="AjayDataLabs | Real Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Real-Time Financial Fraud Detection")
st.markdown("### 🟢 Live System Monitor (Running Real Isolation Forest Model)")

# --- 1. LOAD RESOURCES (CACHED) ---
# We use @st.cache to load the model only once, so it's fast
@st.cache_resource
def load_resources():
    model = joblib.load("models/isolation_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    # Load the sample of the real dataset
    return pd.read_csv("data/sample_stream.csv")

# Load everything
try:
    model, scaler = load_resources()
    stream_data = load_data()
    st.success("✅ Model & Real Data Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading resources: {e}")
    st.stop()

# --- 2. SESSION STATE ---
if "rows_processed" not in st.session_state:
    st.session_state.rows_processed = 0
    st.session_state.history = pd.DataFrame(columns=['timestamp', 'amount', 'anomaly_score', 'prediction'])

# --- 3. CONTROLS ---
st.sidebar.header("🔧 Control Panel")
run_simulation = st.sidebar.checkbox("Run Live Inference", value=True)
speed = st.sidebar.slider("Processing Speed (sec)", 0.0, 1.0, 0.2)

# --- 4. MAIN LOOP ---
placeholder = st.empty()

if run_simulation:
    # Get the next row from the REAL dataset
    if st.session_state.rows_processed < len(stream_data):
        row = stream_data.iloc[st.session_state.rows_processed]
        
        # --- A. PRE-PROCESSING (The "Brain" Work) ---
        # 1. Prepare features (V1..V28 + Amount)
        features = row.drop(['Time', 'Class'], errors='ignore') # Drop non-features if present
        
        # 2. Scale Data (CRITICAL: Must use the same scaler as training)
        # We need to reshape it to look like a dataframe with 1 row
        features_df = pd.DataFrame([features])
        scaled_features = scaler.transform(features_df)
        
        # 3. Predict (The Real Model Decision)
        pred_code = model.predict(scaled_features)[0] # 1 = Normal, -1 = Anomaly
        score = model.decision_function(scaled_features)[0] # Negative = Anomaly
        
        prediction = "Anomaly" if pred_code == -1 else "Normal"
        
        # --- B. UPDATE DISPLAY MEMORY ---
        new_record = {
            "timestamp": time.strftime("%H:%M:%S"),
            "amount": row.get("Amount", 0), # Get Amount safely
            "anomaly_score": score,
            "prediction": prediction
        }
        
        # Append to history
        st.session_state.history = pd.concat([pd.DataFrame([new_record]), st.session_state.history], ignore_index=True).head(200)
        
        # Increment counter
        st.session_state.rows_processed += 1
        
        # --- C. VISUALIZATION ---
        df = st.session_state.history
        anomalies = df[df['prediction'] == 'Anomaly']
        
        with placeholder.container():
            # KPI Row
            k1, k2, k3 = st.columns(3)
            k1.metric("Transactions Scanned", st.session_state.rows_processed)
            k2.metric("🚨 Anomalies Found", len(anomalies), delta_color="inverse")
            
            # Dynamic Risk Status
            risk_level = "LOW"
            if len(anomalies) > 0:
                last_score = df.iloc[0]['anomaly_score']
                risk_level = "CRITICAL" if last_score < -0.1 else "MODERATE"
            k3.metric("Current Risk Level", risk_level)

            # Charts
            c1, c2 = st.columns(2)
            
            # Anomaly Score Trend
            fig_line = px.line(df, x='timestamp', y='anomaly_score', title="Real-Time Model Decision Scores", markers=True)
            fig_line.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Threshold")
            c1.plotly_chart(fig_line, use_container_width=True)
            
            # Amount vs Risk
            fig_scat = px.scatter(df, x='amount', y='anomaly_score', color='prediction',
                                  color_discrete_map={"Normal": "blue", "Anomaly": "red"},
                                  title="Transaction Amount vs. Model Score")
            c2.plotly_chart(fig_scat, use_container_width=True)

            # Data Table
            st.dataframe(df.style.applymap(lambda x: 'color: red; font-weight: bold' if x == 'Anomaly' else 'color: green', subset=['prediction']), use_container_width=True)

    else:
        st.warning("⚠️ End of Sample Data Stream. Refresh page to restart.")

    time.sleep(speed)
    st.rerun()