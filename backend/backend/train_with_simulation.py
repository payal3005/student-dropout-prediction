# train_with_simulation.py
# ONE SCRIPT: training + state-based prediction

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False

# =========================================
# DEBUG: CONFIRM BACKEND FILE IS LOADED
# =========================================
print("🟦 train_with_simulation.py LOADED (v1-debug)")


# ---------------- PATHS ----------------
BACKEND_DIR = "/content/drive/MyDrive/dropout_project/DropOut/backend"
WEBAPP_DIR = "/content/drive/MyDrive/dropout_project/DropOut/webapp"

FEATURES_CSV = f"{BACKEND_DIR}/features_cleaned.csv"
TARGET_CSV = f"{BACKEND_DIR}/target_cleaned.csv"
STATE_WEIGHTS_CSV = f"{BACKEND_DIR}/state_feature_weights.csv"
TEACHERS_CSV = f"{WEBAPP_DIR}/teachers.csv"

MODEL_PKL = f"{BACKEND_DIR}/model.pkl"
SCALER_PKL = f"{BACKEND_DIR}/scaler.pkl"
TRAIN_COLS_PKL = f"{BACKEND_DIR}/train_columns.pkl"
THRESHOLD_PKL = f"{BACKEND_DIR}/threshold.pkl"

RAW_PRED_CSV = f"{BACKEND_DIR}/predictions_raw.csv"
WEIGHTED_PRED_CSV = f"{BACKEND_DIR}/predictions_with_state_weights.csv"

# Top 4 always fixed
TOP_4 = ["failed_courses", "total_participation", "attendance_count", "attendance_rate"]

RANDOM_SEED = 42


# ---------------- Utilities ----------------
def choose_threshold(y_true, y_prob, min_precision=0.1):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    mask = precisions >= min_precision
    if mask.any():
        idx = np.argmax(recalls[mask])
        return float(thresholds[mask][idx])

    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    idx = np.argmax(f1)
    return float(thresholds[idx])

# ======================= TRAIN MODE ============================
def train_model():

    print("\n===================================")
    print("🟦 ENTERED train_model() — TRAINING STARTED (v1-debug)")
    print("===================================")

    X = pd.read_csv(FEATURES_CSV, index_col=0)
    y = pd.read_csv(TARGET_CSV, index_col=0)["dropped"].reindex(X.index).fillna(0).astype(int)

    # Drop ID columns
    for c in ["student_id", "student_pk", "id"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    # log household_income
    if "household_income" in X.columns:
        X["household_income"] = np.log1p(pd.to_numeric(X["household_income"], errors="coerce").fillna(0))

    # SCALE
    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X.index)

    train_cols = X_scaled.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

    # SMOTE
    if HAS_SMOTE and (y_tr.sum() >= 2):
        sm = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
        print("SMOTE applied")
    else:
        X_tr_res, y_tr_res = X_tr, y_tr

    # Train model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )

    print("Training model...")
    model.fit(X_tr_res, y_tr_res)

    # Pick threshold
    y_prob = model.predict_proba(X_te)[:, 1]
    thresh = choose_threshold(y_te.values, y_prob, min_precision=0.1)
    print("Threshold chosen:", thresh)

    # Save artifacts
    with open(MODEL_PKL, "wb") as f: pickle.dump(model, f)
    with open(SCALER_PKL, "wb") as f: pickle.dump(scaler, f)
    with open(TRAIN_COLS_PKL, "wb") as f: pickle.dump(train_cols, f)
    with open(THRESHOLD_PKL, "wb") as f: pickle.dump(thresh, f)

    print("\n🟦 MODEL, SCALER, COLS, THRESHOLD SAVED FROM NEW TRAIN SCRIPT")
    print("🟦 MODEL PATH:", MODEL_PKL)
    print("🟦 SCALER PATH:", SCALER_PKL)
    print("🟦 TRAIN_COLS PATH:", TRAIN_COLS_PKL)
    print("🟦 THRESHOLD PATH:", THRESHOLD_PKL)

    # Save model version stamp
    import json
    stamp = {"model_version": "trained_by_train_with_simulation_v1_debug"}
    STAMP_PATH = os.path.join(BACKEND_DIR, "model_stamp.json")
    with open(STAMP_PATH, "w") as f:
        json.dump(stamp, f)

    print("🟦 Model Stamp Written:", stamp)

    # Save raw predictions
    raw_probs = model.predict_proba(X_scaled)[:, 1]
    raw_pred = (raw_probs >= thresh).astype(int)
    out = X.copy()
    out["dropout_prob_raw"] = raw_probs
    out["predicted_label_raw"] = raw_pred
    out.to_csv(RAW_PRED_CSV)

    print("✔ Training complete.")
    print("Saved raw predictions to:", RAW_PRED_CSV)


# ======================= PREDICT MODE ============================
def predict_with_state():

    print("\n===================================")
    print("🟦 ENTERED predict_with_state() — BACKEND DEBUG")
    print("===================================")

    teacher_username = os.environ.get("TEACHER_USERNAME", None)
    if teacher_username is None:
        teacher_username = input("Enter teacher username: ")

    df = pd.read_csv(TEACHERS_CSV)
    if "state" not in df.columns:
        raise SystemExit("ERROR: teachers.csv missing 'state' column")

    row = df[df["username"] == teacher_username]
    if row.empty:
        raise SystemExit(f"Teacher {teacher_username} not found")

    teacher_state = row.iloc[0]["state"]
    print(f"Teacher {teacher_username} logged in from {teacher_state}")

    weights_df = pd.read_csv(STATE_WEIGHTS_CSV).set_index("state")
    weights_row = weights_df.loc[teacher_state].to_dict()

    with open(MODEL_PKL, "rb") as f: model = pickle.load(f)
    with open(SCALER_PKL, "rb") as f: scaler = pickle.load(f)
    with open(TRAIN_COLS_PKL, "rb") as f: train_cols = pickle.load(f)
    with open(THRESHOLD_PKL, "rb") as f: thresh = pickle.load(f)

    X = pd.read_csv(FEATURES_CSV, index_col=0)

    Xw = X.copy()
    for col, w in weights_row.items():
        if col in Xw.columns and col not in TOP_4:
            try:
                Xw[col] = pd.to_numeric(Xw[col], errors="coerce").fillna(0) * float(w)
            except:
                pass

    X_num = Xw.select_dtypes(include=[np.number]).fillna(0)
    for c in train_cols:
        if c not in X_num.columns:
            X_num[c] = 0.0

    X_scaled = X_num[train_cols]
    X_scaled = pd.DataFrame(scaler.transform(X_scaled), columns=train_cols, index=X.index)

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= thresh).astype(int)

    out = X.copy()
    out["dropout_prob_weighted"] = probs
    out["predicted_label_weighted"] = preds

    out.to_csv(WEIGHTED_PRED_CSV, index=True)
    print("✔ Saved weighted predictions:", WEIGHTED_PRED_CSV)

    SAVE_PATH = f"{BACKEND_DIR}/predictions_state_no_db.csv"
    out.to_csv(SAVE_PATH, index=True)
    print("✔ Saved predictions_state_no_db.csv")
    print("✔ MySQL update skipped (CSV-ONLY mode enabled)")


# ======================= ENTRY POINT ============================
if __name__ == "__main__":

    if "--train" in sys.argv:
        train_model()

    elif "--predict" in sys.argv:
        predict_with_state()

    else:
        print("""
Usage:
  python train_with_simulation.py --train
  python train_with_simulation.py --predict
""")
