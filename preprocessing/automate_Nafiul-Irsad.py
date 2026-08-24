import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_preprocessing(raw_data_path="namadataset_raw/Bitcoin3.csv", output_dir="namadataset_preprocessing"):
    """
    Automates the data cleaning, feature engineering, splitting, and scaling.
    """
    print(f"Loading raw dataset from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    
    # 1. Parse Date and sort
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Handle duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {dup_count} duplicate rows.")
    
    # 3. Handle missing values (forward fill then backward fill for time series)
    df = df.ffill().bfill()
    
    # 4. Feature Engineering: Predict next hour's close price (Target)
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna() # drop last row which has NaN target
    
    # 5. Define features (X) and target (y)
    X = df.drop(columns=['Date', 'Target'])
    y = df['Target']
    
    # 6. Train-test split (80% train, 20% test chronologically)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Dataset split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # 7. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 8. Save output datasets
    os.makedirs(output_dir, exist_ok=True)
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    X_train_df.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"Successfully saved all preprocessed files to: {output_dir}/")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

if __name__ == "__main__":
    # If run directly, run preprocessing with default paths
    # Resolve relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_path = os.path.join(project_root, "namadataset_raw", "Bitcoin3.csv")
    out_dir = os.path.join(script_dir, "namadataset_preprocessing")
    
    run_preprocessing(raw_path, out_dir)
