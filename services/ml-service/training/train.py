import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

from app.config import settings
from training.generate_synthetic_data import generate_synthetic_dataset

def run_training():
    print("=" * 60)
    print("Training PharmaChain Isolation Forest Anomaly Detection Model")
    print("=" * 60)

    # 1. Generate realistic supply chain behavioral dataset
    print("[1/4] Generating synthetic supply chain events (Normal & Anomaly distributions)...")
    df, X, y = generate_synthetic_dataset(n_normal_packs=4000, n_anomalous_packs=200, seed=42)
    print(f"      Total records: {len(X)} | Features: {X.shape[1]}")
    print(f"      Normal samples: {sum(y == 1)} | Injected anomalies: {sum(y == -1)}")

    # 2. Fit Isolation Forest
    print("\n[2/4] Training IsolationForest (n_estimators=200, contamination=0.02)...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    # Train primarily on normal flow
    X_train_normal = X[y == 1]
    model.fit(X_train_normal)
    print("      Model fit complete.")

    # 3. Evaluate model
    print("\n[3/4] Evaluating Anomaly Detection performance...")
    # sklearn IsolationForest outputs 1 for inlier, -1 for outlier
    y_pred = model.predict(X)
    
    # Invert score_samples so higher means more anomalous
    raw_scores = -model.score_samples(X)
    # Target binary labels: 1 for anomaly, 0 for normal
    y_true_binary = (y == -1).astype(int)

    roc_auc = roc_auc_score(y_true_binary, raw_scores)
    print(f"      ROC-AUC Score: {roc_auc:.4f}")
    print("\n      Classification Metrics (Outlier Detection):")
    print(classification_report(y, y_pred, target_names=["Anomaly (-1)", "Normal (+1)"], digits=4))

    # 4. Save Model Artifacts & Baseline Score Distribution (for B9 Drift Monitoring)
    print("[4/4] Serializing model artifacts...")
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = settings.MODEL_PATH
    joblib.dump(model, model_path)
    print(f"      Saved model to: {model_path}")

    # Compute normalized baseline scores on training set
    raw_train_scores = -model.score_samples(X_train_normal)
    min_s, max_s = 0.38, 0.70
    norm_train_scores = np.clip((raw_train_scores - min_s) / (max_s - min_s), 0.0, 1.0)

    dist_path = settings.BASELINE_DIST_PATH
    np.save(dist_path, norm_train_scores)
    print(f"      Saved baseline score distribution ({len(norm_train_scores)} samples) to: {dist_path}")

    print("\n" + "=" * 60)
    print("SUCCESS: Model training & artifact serialization complete!")
    print("=" * 60)

if __name__ == "__main__":
    run_training()
