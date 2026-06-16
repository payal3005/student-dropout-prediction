# 🎓 AI-Based Student Dropout Prediction System

An intelligent machine learning-based web application designed to identify students at risk of dropping out by analyzing academic, attendance, behavioral, and socio-economic factors. The system enables educational institutions to take early intervention measures and improve student retention rates.

---

## Overview

Student dropout is a major challenge faced by educational institutions. This project uses machine learning techniques to predict dropout risk and provide actionable insights through an interactive web dashboard.

The system allows users to:

* Upload student datasets
* Predict dropout risk for individual or multiple students
* Analyze educational indicators
* Generate prediction reports
* Visualize dropout trends and patterns

---

##  Features

* Machine Learning-based dropout prediction
* Interactive dashboard and analytics
* CSV dataset upload support
* Batch and individual student prediction
* Data preprocessing and feature engineering
* Feature importance analysis
* State-wise educational indicator integration
* Downloadable prediction results

---

## Technologies Used

### Backend

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn

### Frontend

* HTML
* CSS
* JavaScript

### Machine Learning

* Classification Models
* Data Preprocessing
* Feature Engineering
* Model Evaluation

---

##  Project Structure

```text
student-dropout-prediction/
│
├── backend/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── predict_dropout.py
│   ├── predict_real.py
│   ├── dashboard_app.py
│   └── supporting datasets and scripts
│
├── webapp/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── uploads/
│   └── uploaded datasets and prediction outputs
│
├── .gitignore
└── README.md
```

---

##  Input Parameters

The model considers multiple educational indicators, including:

* Academic performance
* Attendance records
* Behavioral metrics
* Literacy indicators
* Socio-economic factors
* Institutional statistics
* State-level educational data

---

##  Installation

### Clone the Repository

```bash
git clone https://github.com/payal3005/student-dropout-prediction.git
cd student-dropout-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Navigate to the web application directory:

```bash
cd webapp
```

Run the Flask application:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

##  Prediction Output

The system categorizes students into:

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk

Based on the prediction results, institutions can identify students requiring academic or administrative intervention.

---

##  Project Objectives

* Reduce student dropout rates
* Enable data-driven decision making
* Support early intervention strategies
* Improve educational outcomes
* Assist institutions in student retention planning

---

## Future Enhancements

* Real-time database integration
* Cloud deployment
* User authentication and role management
* Deep learning-based prediction models
* Mobile application support
* Automated intervention recommendations

---

## Author

**Payal A**

BMS Institute of Technology & Management

---

## License

This project is intended for academic and educational purposes.
