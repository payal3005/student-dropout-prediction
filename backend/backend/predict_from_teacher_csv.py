import pandas as pd
import numpy as np
import pickle
import os

# ================================
# PATH CONFIG
# ================================
BACKEND_DIR = "/content/drive/MyDrive/dropout_project/DropOut/backend"
WEBAPP_DIR = "/content/drive/MyDrive/dropout_project/DropOut/webapp"

MODEL_PKL = f"{BACKEND_DIR}/model.pkl"
SCALER_PKL = f"{BACKEND_DIR}/scaler.pkl"
TRAIN_COLS_PKL = f"{BACKEND_DIR}/train_columns.pkl"
THRESHOLD_PKL = f"{BACKEND_DIR}/threshold.pkl"
STATE_WEIGHTS = f"{BACKEND_DIR}/state_feature_weights.csv"
TEACHERS_CSV = f"{WEBAPP_DIR}/teachers.csv"

TOP_4 = ["failed_courses", "total_participation", "attendance_count", "attendance_rate"]


# ====================================================
# MAIN FUNCTION — CONVERTS TEACHER CSV → ML FEATURES
# ====================================================
def predict_from_teacher_csv(csv_path, teacher_username):

    # -------------------------------------
    # 1. Load the teacher-uploaded CSV
    # -------------------------------------
    df = pd.read_csv(csv_path)

    # -------------------------------------
    # 2. Required: gender → gender_M
    # -------------------------------------
    df["gender_M"] = df["gender"].apply(lambda x:
                                        1 if str(x).strip().upper() == "M" else 0)

    # -------------------------------------
    # 3. Add missing engineered fields
    # (since your training data had socioeconomic categories)
    # -------------------------------------
    df["socioeconomic_cat_low"] = 0
    df["socioeconomic_cat_mid"] = 0

    # -------------------------------------
    # 4. Load training column order
    # -------------------------------------
    with open(TRAIN_COLS_PKL, "rb") as f:
        train_cols = pickle.load(f)

    X = df.copy()

    # Ensure all expected columns exist
    for col in train_cols:
        if col not in X.columns:
            X[col] = 0

    # Enforce correct order
    X = X[train_cols].fillna(0)

    # -------------------------------------
    # 5. Load teacher → state
    # -------------------------------------
    teachers_df = pd.read_csv(TEACHERS_CSV)
    teachers_df.columns = teachers_df.columns.str.strip().str.lower()

    teacher_row = teachers_df[teachers_df["username"] == teacher_username]

    if teacher_row.empty:
        raise ValueError(f"Teacher username '{teacher_username}' not found in teachers.csv")

    state = teacher_row.iloc[0]["state"]

    # -------------------------------------
    # 6. Load state weights for that state
    # -------------------------------------
    weights_df = pd.read_csv(STATE_WEIGHTS).set_index("state")

    if state not in weights_df.index:
        raise ValueError(f"State '{state}' not found in state_feature_weights.csv")

    state_weights = weights_df.loc[state].to_dict()

    # -------------------------------------
    # 7. Apply state weights to features
    # -------------------------------------
    X_weighted = X.copy()

    for col, w in state_weights.items():
        if col in X_weighted.columns and col not in TOP_4:
            try:
                X_weighted[col] = pd.to_numeric(X_weighted[col], errors="coerce").fillna(0) * float(w)
            except:
                pass  # ignore non-numeric fields

    # -------------------------------------
    # 8. Scale using saved scaler
    # -------------------------------------
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X_weighted)

    # -------------------------------------
    # 9. Predict using trained model
    # -------------------------------------
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    with open(THRESHOLD_PKL, "rb") as f:
        threshold = pickle.load(f)

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    # -------------------------------------
    # 10. Attach predictions to original dataframe
    # -------------------------------------
    df["dropout_prob"] = probs
    df["predicted_label"] = preds

    return df
