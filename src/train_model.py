import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def train_model():
    print("🧠 Loading Training Data...")
    try:
        df = pd.read_csv("data/train_data.csv")
    except FileNotFoundError:
        print("❌ Error: data/train_data.csv not found. Run data_loader.py first.")
        return

    # We only use features V1-V28 and Amount. 
    # We DROP 'Time' (irrelevant) and 'Class' (because Unsupervised models don't use labels!)
    X = df.drop(columns=['Time', 'Class'])
    
    print("⚖️ Scaling Data (Crucial for Anomaly Detection)...")
    # RobustScaler is best for fraud because it handles outliers better than Standard mean/stdev
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler! We need this to scale the live transactions in the API exactly the same way
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to models/scaler.pkl")

    print("🌲 Training Isolation Forest (This may take 1-2 minutes)...")
    # n_estimators=100: Number of trees
    # contamination=0.002: We estimate ~0.2% of transactions are fraud
    # n_jobs=-1: Use all CPU cores for speed
    model = IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1)
    
    model.fit(X_scaled)
    
    # Save the trained model
    joblib.dump(model, "models/isolation_forest.pkl")
    print("✅ Model saved to models/isolation_forest.pkl")
    print("🚀 Training Complete. We are ready to build the Real-Time API.")

if __name__ == "__main__":
    train_model()