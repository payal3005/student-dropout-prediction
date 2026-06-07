# ===============================================================
# webapp/app.py  (FULL UPDATED VERSION WITH SHAP SUPPORT)
# ===============================================================


import os
import sys
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    session, send_file, url_for, flash, jsonify
)
from werkzeug.utils import secure_filename
import shap


# ============================
# ADD BACKEND PATH
# ============================
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from predict_real import predict_real     # UPDATED SHAP-ENABLED PIPELINE


# ============================
# FLASK SETUP
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = "dropout_secret_key"

@app.route("/__probe")
def probe():
    return "OK FROM FLASK"



# ============================
# UPLOADS FOLDER
# ============================
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================
# TEACHERS CSV
# ============================
teachers_path = os.path.join(os.path.dirname(__file__), "teachers.csv")


# ==========================================================
# GLOBAL STORAGE FOR SHAP
# These will be filled after /predict is called
# ==========================================================
LAST_RESULTS = None
LAST_MODEL = None
LAST_X = None
LAST_FEATURES = None


# ============================
# LOGIN PAGE
# ============================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '').strip()

        df = pd.read_csv(teachers_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.lower()

        df['username'] = df['username'].astype(str).str.strip().str.lower()
        df['password'] = df['password'].astype(str).str.strip()
        df['fullname'] = df['fullname'].astype(str).str.strip()
        df['state'] = df['state'].astype(str).str.strip()

        teacher = df[(df['username'] == username) & (df['password'] == password)]

        if not teacher.empty:
            session['user'] = teacher.iloc[0]['fullname']
            session['username'] = teacher.iloc[0]['username']
            session['state'] = teacher.iloc[0]['state']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password.")

    return render_template('login.html')


# ============================
# DASHBOARD PAGE
# ============================
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html', teacher=session['user'])


# ===============================================================
# 🔥 PREDICT ROUTE — NOW SHAP-ENABLED
# ===============================================================
@app.route('/predict', methods=['POST'])
def predict():
    global LAST_RESULTS, LAST_MODEL, LAST_X, LAST_FEATURES

    if 'user' not in session:
        return redirect('/')

    file = request.files.get('file')
    if not file:
        flash("Please upload a CSV file.", "error")
        return redirect('/dashboard')

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    username = session["username"]

    # --------------------------
    # RUN FULL MODEL PIPELINE
    # predict_real() NOW RETURNS:
    # results, model, X_scaled, train_cols
    # --------------------------
    results, model, X_scaled, train_cols = predict_real(filepath, username)

    # Store for SHAP
    LAST_RESULTS = results
    LAST_MODEL = model
    LAST_X = X_scaled
    LAST_FEATURES = train_cols

    # Save predicted CSV
    output_path = filepath.replace(".csv", "_predicted.csv")
    results.to_csv(output_path, index=False)

    # Clean table for UI
    clean_table_html = (
        results.head(20)
        .to_html(classes='data', header=True, index=False)
        .replace('\n', '')
    )

    # High-risk subset
    at_risk_df = results[results['dropout_prob'] >= 0.3]

    # FOR KPI CARDS
    total_students = len(results)
    high_risk_count = len(at_risk_df)
    avg_grade = round(results["avg_grade"].mean(), 2) if "avg_grade" in results.columns else None
    avg_attendance = round(results["attendance_rate"].mean(), 2) if "attendance_rate" in results.columns else None
    median_dropout = round(results["dropout_prob"].median(), 3)
    high_risk_pct = round((high_risk_count / total_students) * 100, 2)

    return render_template(
        'predict.html',
        teacher=session['user'],
        tables=clean_table_html,
        at_risk_df=at_risk_df.to_dict(orient="records"),
        dropout_probs=results['dropout_prob'].tolist(),

        # KPI values
        total_students=total_students,
        high_risk_count=high_risk_count,
        avg_grade=avg_grade,
        avg_attendance=avg_attendance,
        median_dropout=median_dropout,
        high_risk_pct=f"{high_risk_pct}%",
        
        download_link=url_for('download_file', filename=os.path.basename(output_path))
    )


# ===============================================================
# 🔥 SHAP EXPLANATION ENDPOINT
# ===============================================================
@app.route('/explain/<student_id>')
def explain(student_id):
    global LAST_RESULTS, LAST_MODEL, LAST_X, LAST_FEATURES

    if LAST_RESULTS is None:
        return jsonify({"error": "Please upload a CSV and run prediction first."}), 400

    try:
        student_id = int(student_id)
    except:
        return jsonify({"error": "Invalid student ID"}), 400

    row = LAST_RESULTS[LAST_RESULTS["student_id"] == student_id]
    if row.empty:
        return jsonify({"error": "Student not found"}), 404

    idx = row.index[0]
    x_row = LAST_X.iloc[[idx]]

    # SHAP EXPLAINABILITY
    explainer = shap.TreeExplainer(LAST_MODEL)
    shap_values = explainer.shap_values(x_row)[0]

    shap_list = []
    for f, v in zip(LAST_FEATURES, shap_values):
        shap_list.append({
            "feature": f,
            "value": float(v)
        })

    # sort top features
    shap_list = sorted(shap_list, key=lambda x: abs(x['value']), reverse=True)

    # ====== Simple English Explanation ======
    top3 = shap_list[:3]

    def clean(name):
        return name.replace("_", " ").title()

    conclusion_parts = [
        f"{clean(t['feature'])}: {'increases' if t['value']>0 else 'reduces'} dropout risk"
        for t in top3
    ]
    conclusion = " | ".join(conclusion_parts)

    return jsonify({
        "student_id": student_id,
        "shap": shap_list[:10],
        "conclusion": conclusion
    })


# ============================
# DOWNLOAD FILE
# ============================
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)


# ============================
# LOGOUT
# ============================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# ============================
# RUN SERVER
# ============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
