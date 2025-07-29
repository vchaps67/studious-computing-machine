# studious-computing-machine
Data  Project Portfolio
Welcome to my Data  Project Portfolio! This repository showcases a collection of projects that highlight my skills and expertise in data projects. Each project utilizes various tools and techniques to derive insights from data, solve real-world problems, and support decision-making processes.

Table of Contents

About Me

Projects

Project 1 : IRSF Fraud Detection using Supervised and Unsupervised Machine Learning

Project 2: GPON Fault Detection using Machine Learning

Project 3:Bank Card Fraud Detection Dashboard Project

Project 4: Excel Online Payment Dashboard

Contact

About Me

I am a passionate  about all things data.


Projects

Project 1:  IRSF Fraud Detection using Supervised and Unsupervised Machine Learning

This project simulates and detects International Revenue Share Fraud (IRSF) using synthetic Call Detail Records (CDRs). We explore both supervised and unsupervised learning models to identify anomalous calling behavior and compare their performance.

---

## 📊 Dataset Overview

- **Synthetic CDRs Generated**: 100,000
- **Fields Included**:
  - `A_number`: Originating number
  - `B_number`: Destination number
  - `Duration`: Call duration in seconds
  - `Timestamp`: Date and time of the call
  - `Day`: Day of the week
  - `True_Fraud`: Ground truth fraud label (hidden during unsupervised training)
  - `Manual_Fraud_Tag`: Simulated manual tagging (used in supervised training)

---

## 🛠 Feature Engineering

We created additional features from the original data:
- `Hour`: Extracted from timestamp
- `Is_International`: Flag based on country code difference
- `Is_Weekend`: Based on `Day`
- `Day_encoded`: Numeric encoding of day

---

## 📌 Modeling Approaches

### 1️⃣ Supervised Learning - Random Forest

- **Train/Test Size**: 70/30 split
- **Features Used**: `Duration`, `Hour`, `Is_International`, `Is_Weekend`, `Day_encoded`
- **Evaluation on 30,000 samples**:

| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 80.19%   |
| Precision (Fraud) | 20.2%    |
| Recall (Fraud)    | 24.2%    |
| F1-Score (Fraud)  | 22.0%    |

🔎 Note: Class imbalance impacts precision/recall — further techniques like SMOTE or ensemble methods could improve this.

---

### 2️⃣ Unsupervised Learning - Isolation Forest

- **Entire dataset evaluated (100,000 CDRs)**
- **Anomaly scores used for fraud detection**
- **Compared with hidden `True_Fraud` labels**:

| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 99.94%   |
| Precision (Fraud) | 97.1%    |
| Recall (Fraud)    | 99.8%    |
| F1-Score (Fraud)  | 98.5%    |

✅ Isolation Forest proved highly effective in this synthetic setup.

---

## 📈 Comparison Summary

| Model                         | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|------------------------------|----------|-------------------|----------------|------------------|
| Random Forest (Supervised)   | 80.19%   | 20.2%             | 24.2%          | 22.0%            |
| Isolation Forest (Unsupervised) | 99.94% | 97.1%             | 99.8%          | 98.5%            |

---

## 🔍 Key Takeaways

- **Unsupervised models like Isolation Forest can be extremely effective** in detecting IRSF, especially where labeled data is scarce.
- **Supervised learning benefits from richer features and balanced datasets** — future iterations can explore feature engineering and cost-sensitive learning.
- **This project demonstrates a practical machine learning pipeline** applicable to telecom fraud detection, including data generation, modeling, and evaluation.

---

## 🧠 Technologies Used

- Python (Pandas, NumPy, scikit-learn)
- Matplotlib/Seaborn (optional for visualization)
- Jupyter Notebook for experimentation

---

## 📁 Files

- `irsf_fraud_detection.ipynb`: Full code and explanations
- `cdr_dataset.csv`: Synthetic data (optional)
- `README.md`: This file




---
project 2: GPON Fault Prediction System  
**Proactive Fault Detection in Fiber Access Networks Using Machine Learning**

---

## 📌 Project Overview

This project applies machine learning to predict faults in GPON (Gigabit Passive Optical Network) systems. Using both supervised and unsupervised models, the system identifies high-risk ONTs and recommends proactive interventions. The goal is to reduce downtime, lower operational costs, and enhance customer satisfaction.

---

## ⚙️ System Workflow

```text
1. Synthetic GPON Data Generation (10,000 records)
2. Exploratory Data Analysis
3. Supervised Model Training (XGBoost, RF, etc.)
4. Unsupervised Anomaly Detection (Isolation Forest, DBSCAN)
5. Feature Importance Evaluation
6. Deployment Strategy Formulation
7. Real-Time Monitoring and Alerting Design
8. Cost-Benefit Impact Assessment

📊 Dataset Summary
Total Samples: 10,000 ONT data points

Features: 16 (e.g., TX/RX power, error rate, temperature)

Fault Rate: ~27.3% (binary label)

Train-Test Split: 80% - 20% (8,000 train / 2,000 test)

📈 Model Performance Summary
Model	AUC Score	Accuracy	Fault F1 Score
✅ XGBoost	0.9940	0.97	0.95
Random Forest	0.9918	0.97	0.95
Logistic Regression	0.9814	0.95	0.90
SVM	0.9796	0.95	0.91
Decision Tree	0.9486	0.96	0.93
Isolation Forest	0.7302	0.79	0.42
DBSCAN	0.5000	0.27	0.43

🔍 Top 10 Features Influencing Faults
Feature	Importance
error_rate	31.7%
tx_power_dbm	25.6%
rx_power_dbm	18.4%
power_budget_db	13.0%
snr_estimate_db	5.3%
ont_age_years	0.94%
bend_loss_db	0.92%
splitter_loss_db	0.91%
traffic_utilization_pct	0.86%
temperature_c	0.83%

🚀 Deployment Strategy
Primary Model
XGBoost – Best performing supervised model (AUC: 0.994)

Integrated with Isolation Forest for anomaly detection

Thresholds
High Risk (Red): Probability > 0.8 → Immediate Action

Medium Risk (Yellow): 0.5–0.8 → Maintenance Scheduling

Low Risk (Green): 0.2–0.5 → Monitor

🔧 Monitoring & Alerting
KPI Triggers:

RX Power < -25 dBm

Error Rate > 0.01

Power Budget Deviation

Extreme Temperature Shifts

Alert Levels:

Level 1: Automated diagnostics

Level 2: Predictive technician dispatch

Level 3: Emergency response

🛠 Preventive Maintenance Plan
Risk Level	Inspection Frequency	Action Items
High Risk	Monthly	ONT replacement, fiber cleaning
Medium Risk	Quarterly	Signal integrity checks
Low Risk	Annually	Environmental audit, documentation review

--





Project 1: Payment Fraud Detection

Description: This project is about detection on payment fraud using a number of models. Intial data was lablled with no observable outliers. The aim of the project is to increase the time to detect fraud and thus reduce the impcat of the fraud. Its aslo tries to enhace the develdopment of fraud prevention strategies.  There were no observable fraud patterns from the data set. However the decstion tree and ensemble methods perfommed well, but oulier methods like IsolationForest performed poorly

Tools Used:  Python.

Key Findings: Detection of novel fraud methods for dataset with little or no obervable outliers is a problem

Link: https://github.com/vchaps67/studious-computing-machine/blob/main/payment_fraud.ipynb

Project 2: Bank Card Fraud Dashboard

Description :  Developed an interactive fraud detection dashboard to monitor and analyze credit card transactions, enabling real-time fraud monitoring, fraud status at a particular point in time  and pattern recognition. 

Tools: Tableau, Python

Key Objectives
Monitor fraud patterns and trends in real-time
Identify high-risk transaction patterns
Track geographical distribution of fraudulent activities
Analyze customer segments and fraud occurrence
Measure and visualize key fraud metrics

Link: https://public.tableau.com/authoring/CreditCardFraudDashboard_17311558849130/Dashboard1#1


Project 3: Excel Online Payment Dashboard

Description:  Develped an interactive Excel dashboard to provide strategic fraud insights over time

Tools: Excel

Key Objectives

Provide a strategice view of online payment fraud over time

Link : https://github.com/vchaps67/studious-computing-machine/commit/5c1ccbcb86872984e15193784f5601e62cdd35ef
