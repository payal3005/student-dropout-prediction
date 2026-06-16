import pandas as pd
import numpy as np
import os
import pickle

# Paths
BACKEND_DIR = os.path.dirname(__file__)
MODEL_PKL = os.path.join(BACKEND_DIR, "model.pkl")
SCALER_PKL = os.path.join(BACKEND_DIR, "scaler.pkl")
TRAIN_COLS_PKL = os.path.join(BACKEND_DIR, "train_columns.pkl")
THRESHOLD_PKL = os.path.join(BACKEND_DIR, "threshold.pkl")
STATE_WEIGHTS_CSV = os.path.join(BACKEND_DIR, "state_feature_weights.csv")
TEACHERS_CSV = os.path.join(os.path.dirname(BACKEND_DIR), "webapp", "teachers.csv")

TOP_4 = ["failed_courses", "total_participation", "attendance_count", "attendance_rate"]


def predict_dropout(file_path, teacher_username, teacher_state):
    """
    Real ML-based dropout prediction with state weighting.
    """

    # --------------------------
    # 1. LOAD STUDENT CSV
    # --------------------------
    df = pd.read_csv(file_path)

    # --------------------------
    # 2. Basic Cleaning
    # --------------------------
    df.columns = df.columns.str.strip().str.lower()

    # Create gender_M
    if "gender" in df.columns:
        df["gender_m"] = (df["gender"].str.upper() == "M").astype(int)

    # Forced missing columns to prevent ML crash
    required_cols = [
        "failed_courses","total_participation","attendance_count","attendance_rate",
        "avg_grade","total_credits","pass_rate","grade_trend",
        "events_attended","first_gen","gender_m","household_income"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # --------------------------
    # 3. Load model artifacts
    # --------------------------
    with open(MODEL_PKL, "rb") as f: model = pickle.load(f)
    with open(SCALER_PKL, "rb") as f: scaler = pickle.load(f)
    with open(TRAIN_COLS_PKL, "rb") as f: train_cols = pickle.load(f)
    with open(THRESHOLD_PKL, "rb") as f: threshold = pickle.load(f)

    # --------------------------
    # 4. Load State Weights
    # --------------------------
    weights_df = pd.read_csv(STATE_WEIGHTS_CSV).set_index("state")
    state_weights = weights_df.loc[teacher_state].to_dict()

    # --------------------------
    # 5. Apply State Weights
    # --------------------------
    df_w = df.copy()

    for col, w in state_weights.items():
        if col in df_w.columns and col not in TOP_4:
            try:
                df_w[col] = df_w[col].astype(float) * float(w)
            except:
                pass

    # --------------------------
    # 6. Scale to Training Columns
    # --------------------------
    X = df_w.reindex(columns=train_cols, fill_value=0)
    X_scaled = scaler.transform(X)

    # --------------------------
    # 7. Predict
    # --------------------------
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["dropout_prob"] = probs
    df["predicted_label"] = preds

    return df
