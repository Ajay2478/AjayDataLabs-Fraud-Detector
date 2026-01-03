import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath="data/creditcard.csv"):
    """
    Loads the European Credit Card dataset and splits it:
    - 80% for Training the Model (Historical data)
    - 20% for Streaming (Simulating live production data)
    """
    print("⏳ Loading dataset... (this might take a few seconds)")
    
    # Load Data
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        print("👉 Please download 'creditcard.csv' and place it in the 'data/' folder.")
        return None, None

    # We only need the features and the label
    # Features: Time, V1...V28, Amount
    # Label: Class (0 = Normal, 1 = Fraud)
    
    print(f"✅ Data Loaded. Shape: {df.shape}")
    
    # Shuffle and Split
    # random_state=42 ensures we get the same split every time (reproducibility)
    train_df, stream_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"🔹 Training Set: {len(train_df)} records (Used to train Isolation Forest)")
    print(f"🔹 Streaming Set: {len(stream_df)} records (Used to simulate live API calls)")
    
    # Save them separately so other scripts can access them easily
    train_df.to_csv("data/train_data.csv", index=False)
    stream_df.to_csv("data/stream_data.csv", index=False)
    
    print("✅ Split complete! Saved 'train_data.csv' and 'stream_data.csv' in data/ folder.")
    return train_df, stream_df

if __name__ == "__main__":
    load_and_split_data()