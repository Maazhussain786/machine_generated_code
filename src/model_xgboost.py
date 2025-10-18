# src/model_xgboost.py
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from src.features_ast import extract_features_df


def train_xgboost(train_df, dev_df, out_dir="experiments/xgboost_task_a"):
    """
    Train an XGBoost model on AST-based structural code features with caching.
    - If cached features exist, loads them directly (fast).
    - Else, extracts features and saves for future runs.
    - Trains and saves XGBoost model + metrics.
    """

    os.makedirs(out_dir, exist_ok=True)


    train_feat_path = os.path.join(out_dir, "X_train_features.joblib")
    dev_feat_path = os.path.join(out_dir, "X_dev_features.joblib")
    train_label_path = os.path.join(out_dir, "y_train.joblib")
    dev_label_path = os.path.join(out_dir, "y_dev.joblib")


    if all(os.path.exists(p) for p in [train_feat_path, dev_feat_path, train_label_path, dev_label_path]):
        print("📂 Loading cached features for XGBoost...")
        X_train = joblib.load(train_feat_path)
        X_dev = joblib.load(dev_feat_path)
        y_train = joblib.load(train_label_path)
        y_dev = joblib.load(dev_label_path)
    else:
        print("🧠 Extracting structural features (train)...")
        X_train, y_train = extract_features_df(train_df, code_col="code", lang_col="language")

        print("🧠 Extracting structural features (dev)...")
        X_dev, y_dev = extract_features_df(dev_df, code_col="code", lang_col="language")

        print(f"✅ Feature shape: train={X_train.shape}, dev={X_dev.shape}")

        # Drop non-numeric columns (like 'lang')
        X_train = X_train.select_dtypes(exclude=["object"])
        X_dev = X_dev.select_dtypes(exclude=["object"])

        # Align columns between train and dev
        X_train, X_dev = X_train.align(X_dev, join="left", axis=1, fill_value=0)

        # Ensure numeric dtype and no NaN
        X_train = X_train.astype(float).fillna(0.0)
        X_dev = X_dev.astype(float).fillna(0.0)
        y_train = y_train.astype(int)
        y_dev = y_dev.astype(int)

        # Save to cache for next run
        joblib.dump(X_train, train_feat_path)
        joblib.dump(X_dev, dev_feat_path)
        joblib.dump(y_train, train_label_path)
        joblib.dump(y_dev, dev_label_path)
        print("💾 Features cached.")


    # Train XGBoost

    dtrain = xgb.DMatrix(X_train, label=y_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",      
        "learning_rate": 0.1,
        "max_depth": 6,
        "seed": 42,
    }

    print("🚀 Training XGBoost for 200 boosting rounds...")
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (ddev, "dev")],
        verbose_eval=50
    )


    model_path = os.path.join(out_dir, "xgboost_model.json")
    booster.save_model(model_path)
    print(f"✅ Model saved to {model_path}")


    artifact_flag = os.path.join(out_dir, "model.joblib")
    joblib.dump({"model_path": model_path}, artifact_flag)


    preds = (booster.predict(ddev) > 0.5).astype(int)
    acc = accuracy_score(y_dev, preds)
    f1 = f1_score(y_dev, preds, average="macro")

    metrics = {"accuracy": float(acc), "f1_macro": float(f1)}
    joblib.dump(metrics, os.path.join(out_dir, "metrics.joblib"))
    print(f"📊 Dev Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"📁 Results saved to: {out_dir}")