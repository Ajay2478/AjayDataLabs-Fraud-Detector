import streamlit as st
import pandas as pd
import sqlite3
import time
import plotly.express as px

# Page Config (Browser Title)
st.set_page_config(
    page_title="AjayDataLabs | Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
)

# Dashboard Title
st.title("🛡️ Real-Time Financial Fraud Detection")
st.markdown(f"**Live System Status:** 🟢 Online | **Engine:** Isolation Forest")

# Connect to Database
DB_NAME = "fraud_detection.db"

def load_data():
    """Fetch the last 100 transactions from the database"""
    conn = sqlite3.connect(DB_NAME)
    # Get last 200 rows to show trends
    df = pd.read_sql("SELECT * FROM transactions ORDER BY id DESC LIMIT 200", conn)
    conn.close()
    return df

# Create placeholders for live updates
kpi_placeholder = st.empty()
charts_placeholder = st.empty()
table_placeholder = st.empty()

# Auto-Refresh Loop
while True:
    # 1. Fetch Data
    df = load_data()
    
    if not df.empty:
        # Calculate Metrics
        total_tx = len(df)
        fraud_tx = df[df['is_fraud'] == 1]
        fraud_count = len(fraud_tx)
        fraud_rate = (fraud_count / total_tx) * 100 if total_tx > 0 else 0
        
        # 2. Update KPIs (Top Row)
        with kpi_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("📦 Recent Transactions", total_tx)
            col2.metric("🚨 Anomalies Detected", fraud_count, delta_color="inverse")
            col3.metric("⚠️ Fraud Rate (Last 200)", f"{fraud_rate:.1f}%")

        # 3. Update Charts (Middle Row)
        with charts_placeholder.container():
            col_left, col_right = st.columns(2)
            
            # Chart 1: Anomaly Score Trend (The "Heartbeat")
            fig_score = px.line(df, x=df.index, y="anomaly_score", 
                                title="Live Anomaly Scores (Lower = More Suspicious)",
                                markers=True)
            # Add a red line for the threshold (approx -0.05 to 0 usually)
            fig_score.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Threshold")
            col_left.plotly_chart(fig_score, use_container_width=True)
            
            # Chart 2: Amount vs. Anomaly (Scatter)
            # Fraud often happens at high amounts or specific low amounts
            fig_scatter = px.scatter(df, x="amount", y="anomaly_score", 
                                     color="prediction", 
                                     title="Transaction Amount vs. Anomaly Score",
                                     color_discrete_map={"Normal": "blue", "Anomaly": "red"})
            col_right.plotly_chart(fig_scatter, use_container_width=True)

        # 4. Update Table (Bottom Row)
        with table_placeholder.container():
            st.subheader("📋 Recent Activity Feed")
            # Show top 10 most recent
            display_df = df[['timestamp', 'amount', 'prediction', 'anomaly_score']].head(10)
            
            # Style the table: Highlight Anomalies in Red
            def highlight_fraud(val):
                color = 'red' if val == 'Anomaly' else 'green'
                return f'color: {color}; font-weight: bold'

            st.dataframe(display_df.style.applymap(highlight_fraud, subset=['prediction']), use_container_width=True)

    # Refresh every 1 second
    time.sleep(1)