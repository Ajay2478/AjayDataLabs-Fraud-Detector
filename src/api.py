from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from src.db import init_db, save_transaction  # <--- NEW IMPORT

# 1. Initialize App & DB
app = FastAPI(title="AjayDataLabs Fraud Detector", version="1.0")

@app.on_event("startup")
def startup_event():
    init_db()  # <--- Initialize DB on startup

# 2. Load Models
print("⏳ Loading model and scaler...")
model = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ Systems Online.")

# 3. Define Input
class Transaction(BaseModel):
    features: dict 

# 4. Prediction Endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.features])
        
        # Cleanup
        if 'Time' in df.columns: df = df.drop(columns=['Time'])
        if 'Class' in df.columns: df = df.drop(columns=['Class'])

        # Inference
        data_scaled = scaler.transform(df)
        prediction = model.predict(data_scaled)[0]
        score = model.decision_function(data_scaled)[0]
        
        result = "Anomaly" if prediction == -1 else "Normal"
        is_fraud = bool(prediction == -1)

        # 5. SAVE TO DB (The New Part) 💾
        # We need the amount for the dashboard visuals
        amount = df['Amount'].values[0]
        save_transaction(amount, float(score), result, is_fraud)

        return {
            "prediction": result,
            "anomaly_score": float(score),
            "is_fraud": is_fraud
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))