# backend/predict_real.py

import os
import pickle
import pandas as pd
import numpy as np

print("🟩 LOADED FILE: predict_real.py (SHAP-ENABLED + SAFE)")
VERSION = "v6.0-shap-safe"
print("🟩 Backend Version:", VERSION)

# ================================
# PATHS
# ================================
BACKEND_DIR = os.path.dirname(__file__)
WEBAPP_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "webapp")

MODEL_PKL = os.path.join(BACKEND_DIR, "model.pkl")
SCALER_PKL = os.path.join(BACKEND_DIR, "scaler.pkl")
TRAIN_COLS_PKL = os.path.join(BACKEND_DIR, "train_columns.pkl")
THRESHOLD_PKL = os.path.join(BACKEND_DIR, "threshold.pkl")
STATE_WEIGHTS_CSV = os.path.join(BACKEND_DIR, "state_feature_weights.csv")
TEACHERS_CSV = os.path.join(WEBAPP_DIR, "teachers.csv")

# columns that should NOT be state-weighted
TOP_4 = ["failed_courses", "total_participation", "attendance_count", "attendance_rate"]


def _safe_to_numeric_df(df, cols):
    """
    Coerce listed columns of df to numeric safely. Returns a tuple:
    (df_converted, list_of_columns_that_had_non_numeric_values)
    """
    non_numeric_cols = []
    for c in cols:
        # If column already numeric dtype, continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        # Attempt coercion
        coerced = pd.to_numeric(df[c], errors="coerce")
        # If any values became NaN where they previously were non-null strings, mark it
        had_non_numeric = coerced.isna() & df[c].notna()
        if had_non_numeric.any():
            non_numeric_cols.append(c)
        df[c] = coerced.fillna(0.0)
    return df, non_numeric_cols


# ===================================================
# MAIN PREDICTION PIPELINE
# ===================================================
def predict_real(uploaded_csv_path: str, teacher_username: str):

    print("\n==============================")
    print("🟩 ENTERED predict_real() — SHAP SAFE + NUMERIC COERCION")
    print("==============================")
    print("CSV:", uploaded_csv_path)
    print("Teacher:", teacher_username)

    # ------------------------------------
    # 1) LOAD CSV
    # ------------------------------------
    df = pd.read_csv(uploaded_csv_path)
    orig = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # ------------------------------------
    # 2) GENDER ENCODING
    # ------------------------------------
    if "gender" in df.columns:
        df["gender_m"] = df["gender"].astype(str).str.upper().map(lambda x: 1 if x == "M" else 0)
    else:
        df["gender_m"] = 0

    # socio cat placeholders
    df["socioeconomic_cat_low"] = 0
    df["socioeconomic_cat_mid"] = 0

    # ------------------------------------
    # 3) INCOME CLEANING
    # ------------------------------------
    if "household_income" in df.columns:
        df["household_income"] = np.log1p(pd.to_numeric(df["household_income"], errors="coerce").fillna(0.0))
    else:
        df["household_income"] = 0.0

    # ------------------------------------
    # 4) LOAD TRAIN ARTIFACTS
    # ------------------------------------
    train_cols = pickle.load(open(TRAIN_COLS_PKL, "rb"))
    scaler = pickle.load(open(SCALER_PKL, "rb"))
    model = pickle.load(open(MODEL_PKL, "rb"))
    threshold = pickle.load(open(THRESHOLD_PKL, "rb"))

    # ------------------------------------
    # 5) ENSURE ALL TRAIN COLS EXIST
    # ------------------------------------
    X = df.copy()
    for col in train_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[train_cols].fillna(0)

    # ------------------------------------
    # 6) TEACHER → STATE
    # ------------------------------------
    teachers_df = pd.read_csv(TEACHERS_CSV)
    teachers_df.columns = teachers_df.columns.str.strip().str.lower()

    teacher_row = teachers_df[teachers_df["username"] == teacher_username]
    if teacher_row.empty:
        raise ValueError(f"Teacher username '{teacher_username}' not found in teachers.csv")
    teacher_state = teacher_row.iloc[0]["state"]

    # ------------------------------------
    # 7) STATE WEIGHTS
    # ------------------------------------
    weights_df = pd.read_csv(STATE_WEIGHTS_CSV).set_index("state")
    if teacher_state not in weights_df.index:
        raise ValueError(f"State '{teacher_state}' not found in state_feature_weights.csv")
    state_weights = weights_df.loc[teacher_state].to_dict()

    # ------------------------------------
    # 8) APPLY SAFE STATE WEIGHTS
    # ------------------------------------
    Xw = X.copy()
    for col, weight in state_weights.items():

        # Skip columns not in data or top-4
        if col not in Xw.columns or col in TOP_4:
            continue

        # Skip non-numeric columns safely
        if not pd.api.types.is_numeric_dtype(Xw[col]):
            # Try to coerce if it looks numeric-ish, else skip
            coerced = pd.to_numeric(Xw[col], errors="coerce")
            if coerced.notna().any():
                Xw[col] = coerced.fillna(0.0) * float(weight)
            else:
                # Column contains strings like names — skip weighting
                print(f"⚠️ Skipping non-numeric column during weighting: {col}")
                continue
        else:
            Xw[col] = pd.to_numeric(Xw[col], errors="coerce").fillna(0.0) * float(weight)

    # ------------------------------------
    # 9) ENSURE ALL COLUMNS ARE NUMERIC BEFORE SCALING
    # ------------------------------------
    # Coerce all train columns to numeric (safe), and report any columns that had non-numeric values
    Xw, non_numeric_columns = _safe_to_numeric_df(Xw, train_cols)
    if non_numeric_columns:
        print("⚠️ The following train columns had non-numeric values and were coerced to 0.0:", non_numeric_columns)

    # ------------------------------------
    # 10) SCALE FEATURES
    # ------------------------------------
    X_scaled = pd.DataFrame(scaler.transform(Xw), columns=train_cols, index=Xw.index)

    # ------------------------------------
    # 11) PREDICT
    # ------------------------------------
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    orig["dropout_prob"] = probs
    orig["predicted_label"] = preds

    # Add student_id if missing
    if "student_id" not in orig.columns:
        orig["student_id"] = orig.index

    print("🟩 Prediction Done.")
    print("🟩 Returning: results, model, X_scaled, train_cols")
    print("==============================\n")

    return orig, model, X_scaled, train_cols
