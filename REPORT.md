# 📑 Project Report: Real-Time Financial Fraud Detection System
**Author:** Ajay (AjayDataLabs)  
**Date:** January 8, 2026  
**Status:** Deployed (v1.0)

---

## 1. Executive Summary
Financial fraud is a critical issue where delayed detection results in massive monetary losses. Traditional systems often rely on batch processing, identifying fraud hours or days after the event.

This project implements a **Real-Time Anomaly Detection Engine** capable of processing financial transactions in milliseconds. By leveraging Unsupervised Machine Learning (**Isolation Forest**), the system detects novel fraud patterns without requiring pre-labeled training data. The final deployed model achieved a **Recall of 84%**, successfully intercepting the majority of fraudulent attempts in the test environment.

---

## 2. Problem Statement
* **Objective:** Identify fraudulent credit card transactions instantly upon occurrence.
* **Challenge:**
    * **Imbalanced Data:** Fraud accounts for only 0.17% of transactions.
    * **Latency:** Decisions must be made in under 100ms to avoid user friction.
    * **Adaptability:** Fraud patterns change rapidly; supervised rules become obsolete quickly.

---

## 3. System Architecture
The solution was engineered as a microservices-style pipeline:

1.  **Ingestion Layer:** A transaction stream simulator replay historical credit card data to mimic a live payment gateway.
2.  **Processing Layer (API):** A **FastAPI** server receives JSON payloads, preprocesses features, and requests model inference.
3.  **Intelligence Layer:** An **Isolation Forest** model (Scikit-Learn) tuned for high recall (Aggressive Anomaly Detection).
4.  **Storage Layer:** **SQLite** database logs every transaction ID, timestamp, amount, and anomaly score for audit trails.
5.  **Presentation Layer:** A **Streamlit** dashboard provides a live command center for security analysts to monitor threats.

---

## 4. Methodology

### 4.1 Data Preprocessing
* **Dataset:** European Credit Card Fraud Dataset (284,807 transactions).
* **Feature Engineering:**
    * Time-agnostic approach (Time column dropped to prevent overfitting to specific hours).
    * **RobustScaler** applied to handle extreme outliers in the 'Amount' feature, which is typical in financial data.

### 4.2 Model Selection
* **Algorithm:** Isolation Forest (Unsupervised).
* **Rationale:** Unlike classification algorithms (Random Forest/XGBoost) which require "Fraud" labels to learn, Isolation Forest isolates anomalies by randomly partitioning data. This makes it superior for detecting *new, unseen* types of fraud.

### 4.3 Hyperparameter Tuning
* **Initial Strategy:** Default parameters (`contamination='auto'`) resulted in low Recall (~27%), missing too many real fraud cases.
* **Optimized Strategy:** Adjusted `contamination=0.05`. This "Aggressive Mode" forces the model to treat the top 5% of outliers as potential fraud.
* **Outcome:** Drastic improvement in fraud detection rate.

---

## 5. Performance Evaluation
The model was tested against a holdout set of **56,962 transactions**.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 95.04% | *Misleading metric due to class imbalance.* |
| **Precision (Fraud)** | 3.00% | *Low precision accepted to ensure maximum safety.* |
| **Recall (Fraud)** | **84.00%** | *The critical success metric. We caught 84% of attacks.* |
| **Latency** | < 50ms | *Average API response time per transaction.* |

### Business Impact Analysis
* **Fraud Caught (True Positives):** 82 transactions.
* **Fraud Missed (False Negatives):** 16 transactions.
* **False Alarms:** 2,807 transactions.
* **Estimated Savings:** Assuming an average fraud loss of $5,000, intercepting 82 attacks potentially saves **$410,000**. The cost of verifying false alarms (SMS/Email) is negligible compared to these savings.

---

## 6. Technical Stack
* **Language:** Python 3.9+
* **ML Libraries:** Scikit-Learn, Joblib, Pandas, NumPy.
* **Backend:** FastAPI, Uvicorn.
* **Frontend:** Streamlit, Plotly (for interactive visualization).
* **Database:** SQLite3.
* **Version Control:** Git / GitHub.

---

## 7. Conclusion & Future Scope
The project successfully demonstrates an end-to-end MLOps workflow. The system is currently live on **Streamlit Cloud**, processing simulated real-world data streams.

**Future Improvements:**
* **Dockerization:** Containerizing the API for scalable cloud deployment (AWS ECS).
* **Feedback Loop:** Implementing a "Mark as False Positive" button in the dashboard to retrain the model periodically.
* **Deep Learning:** Experimenting with Autoencoders (LSTM) for sequence-based anomaly detection.