import joblib
import numpy as np
import pandas as pd

# Load artifacts
MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"

def predict_logic(income, cost, age=45, bedrooms=3):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Model or Scaler file not found")
        return None

    # 1. Safe Calculation
    burden = (cost * 12) / max(income, 1)
    
    # 2. Labeling
    if burden > 2.0:
        label = "Extreme burden"
    elif burden > 0.5:
        label = "Severely burdened"
    elif burden > 0.3:
        label = "Moderately burdened"
    else:
        label = "Affordable"
        
    # 3. Anomaly Detection
    anomalies = []
    if burden > 2.0:
        anomalies.append("Extremely high cost burden (>200%)")
    if income <= 6000:
        anomalies.append("Extremely low income (<= $6000)")
        
    anomaly_flag = len(anomalies) > 0
    
    # Feature Vector
    status_input = 1 if burden < 0.3 else (2 if burden <= 0.5 else 3)
    features = [
        income, cost, 0, 1000, burden, status_input, 
        age, 2, bedrooms, 5, 1, 2, 1, 1, 4
    ]
    
    # Scale & Predict
    features_scaled = scaler.transform([features])
    cluster = int(model.predict(features_scaled)[0])
    
    return {
        "cluster": cluster,
        "cost_burden_ratio": burden,
        "cost_burden_label": label,
        "anomaly_flag": anomaly_flag,
        "anomaly_reasons": anomalies
    }

def run_tests():
    print("--- Running Tests ---")
    
    # Test 1: Extreme Case (Income=1300, Cost=800)
    # Expected: Burden ~7.38, Label=Extreme, Anomaly=True
    res1 = predict_logic(1300, 800)
    print(f"\nTest 1 (Extreme): {res1}")
    if res1['anomaly_flag'] and res1['cost_burden_label'] == "Extreme burden" and res1['cost_burden_ratio'] > 7.0:
        print("PASS")
    else:
        print("FAIL")

    # Test 2: Normal Case (Income=45000, Cost=1200)
    # Expected: Burden ~0.32, Label=Moderately, Anomaly=False, Cluster=0
    res2 = predict_logic(45000, 1200)
    print(f"\nTest 2 (Normal): {res2}")
    if not res2['anomaly_flag'] and res2['cost_burden_label'] == "Moderately burdened" and res2['cluster'] == 0:
        print("PASS")
    else:
        print("FAIL")

    # Test 3: Zero Income (Income=0, Cost=1000)
    # Expected: Burden > 0 (handled by max(1)), Anomaly=True (Low Income + High Burden)
    res3 = predict_logic(0, 1000)
    print(f"\nTest 3 (Zero Income): {res3}")
    if res3['anomaly_flag'] and "Extremely low income (<= $6000)" in res3['anomaly_reasons']:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    run_tests()
