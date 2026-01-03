import pandas as pd
import requests
import time
import random
import json

# API Endpoint
URL = "http://127.0.0.1:8000/predict"

def run_simulation():
    print("🌊 Loading Stream Data...")
    try:
        # Load the "Future" data we saved earlier
        df = pd.read_csv("data/stream_data.csv")
    except FileNotFoundError:
        print("❌ Error: data/stream_data.csv not found.")
        return

    print(f"🚀 Starting Real-Time Simulation with {len(df)} transactions...")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    # Shuffle data to make it unpredictable
    df = df.sample(frac=1).reset_index(drop=True)

    # Loop through transactions
    for index, row in df.iterrows():
        # 1. Prepare the Payload
        # We convert the row to a dictionary
        transaction_dict = row.to_dict()
        
        # Remove 'Class' (The answer) and 'Time' before sending
        # In real life, the payment gateway doesn't know if it's fraud yet!
        true_label = transaction_dict.pop("Class", None) # Keep for our reference
        transaction_dict.pop("Time", None)

        payload = {"features": transaction_dict}

        try:
            # 2. Send Request to API
            start_time = time.time()
            response = requests.post(URL, json=payload)
            latency = (time.time() - start_time) * 1000 # ms
            
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                score = result["anomaly_score"]
                
                # 3. Print Result (Color Coded)
                # If Anomaly -> RED, If Normal -> GREEN
                if prediction == "Anomaly":
                    print(f"🚨 ALERT! [ID: {index}] | Score: {score:.4f} | Latency: {latency:.1f}ms")
                else:
                    # Print normal transactions less frequently to reduce noise, or print all succinctly
                    print(f"✅ Normal  [ID: {index}] | Score: {score:.4f} | Latency: {latency:.1f}ms")
            
            else:
                print(f"❌ Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"⚠️ Connection Failed: {e}")
            break
        
        # 4. Sleep to simulate real-time traffic
        # Random delay between 0.1s and 1.0s
        time.sleep(random.uniform(0.1, 0.5))

if __name__ == "__main__":
    run_simulation()