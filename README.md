# 🛡️ Real-Time Financial Fraud Detection System
### By AjayDataLabs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLOps](https://img.shields.io/badge/MLOps-FastAPI%2BStreamlit-green)
![Status](https://img.shields.io/badge/Status-Live-success)

## 📌 Project Overview
Traditional fraud detection systems run in batches (overnight), allowing fraudsters to drain accounts before they are caught. 
**This project is a Real-Time Anomaly Detection Engine** that processes financial transactions in milliseconds, flagging suspicious activity instantly.

It simulates a live payment gateway, scores transactions using an **Isolation Forest** machine learning model, and visualizes threats on a live dashboard.

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[Stream Simulator] -->|JSON Transaction| B(FastAPI Endpoint)
    B -->|Pre-processing| C{Isolation Forest Model}
    C -->|Normal| D[Log to DB]
    C -->|Anomaly| E[🚨 Trigger Alert]
    E --> D
    D --> F[Streamlit Dashboard]

🛠️ Tech Stack
Core: Python 3.9+

Machine Learning: Scikit-Learn (Isolation Forest), Joblib

API: FastAPI (for real-time inference)

Database: SQLite (lightweight transactional storage)

Dashboard: Streamlit & Plotly (Live visualization)

Container/Deployment: Docker / Streamlit Cloud