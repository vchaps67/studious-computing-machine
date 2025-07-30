# studious-computing-machine
Data  Project Portfolio
Welcome to my Data  Project Portfolio! This repository showcases a collection of projects that highlight my skills and expertise in data projects. Each project utilizes various tools and techniques to derive insights from data, solve real-world problems, and support decision-making processes.

Table of Contents

About Me

Projects

Project 1 : IRSF Fraud Detection using Supervised and Unsupervised Machine Learning

Project 2: GPON Fault Detection using Machine Learning

Project 3: Phishing Website Detector

Project 4: DNS Tunneling and DGA Detection Project



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

Project 3: Phishing Website Detection System
A machine learning-powered system to detect phishing websites using URL analysis and metadata features. This project implements multiple ML models and provides both a command-line interface and an interactive Streamlit dashboard for real-time phishing detection.
🎯 Problem Statement
Phishing attacks continue to be one of the most prevalent cybersecurity threats, with over 1.2 million phishing websites created monthly. Traditional blacklist-based approaches are reactive and easily bypassed. This project develops a proactive ML-based detection system that analyzes URL characteristics and website metadata to identify potential phishing sites in real-time.
🏗️ Architecture Overview
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   URL Input     │───▶│  Feature Engine  │───▶│   ML Models    │
│                 │    │                  │    │                 │
│ • User Input    │    │ • URL Analysis   │    │ • Random Forest │
│ • Browser Plugin│    │ • Domain Check   │    │ • Gradient Boost│
│ • Batch Files   │    │ • WHOIS Data     │    │ • Neural Network│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │                        │
                                 ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Feature Vector   │    │   Prediction    │
                       │                  │    │                 │
                       │ • URL Features   │    │ • Risk Score    │
                       │ • Domain Features│    │ • Classification │
                       │ • Content Features│    │ • Confidence    │
                       └──────────────────┘    └─────────────────┘
📊 Dataset & Data Sources
Primary Data Sources

PhishTank: Real-time phishing URL database
OpenPhish: Community-driven phishing intelligence
Alexa Top 1M: Legitimate website samples
Custom Web Scraping: Additional legitimate URLs

Dataset Statistics

Training Data: ~100,000 URLs (50% phishing, 50% legitimate)
Features: 30+ engineered features
Update Frequency: Daily refreshes from live feeds

🔧 Feature Engineering
URL-Based Features (20 features)
python# URL Structure Analysis
- url_length: Length of the URL
- num_dots: Number of dots in URL
- num_hyphens: Number of hyphens
- num_underscores: Number of underscores
- num_slashes: Number of forward slashes
- num_questionmarks: Number of question marks
- num_equals: Number of equal signs
- num_ats: Number of @ symbols
- num_ampersands: Number of & symbols
- num_exclamations: Number of exclamation marks
- num_spaces: Number of spaces (encoded or not)
- num_tildes: Number of tildes
- num_commas: Number of commas
- num_semicolons: Number of semicolons
- num_dollars: Number of dollar signs
- num_percentages: Number of percentage signs
- shortening_service: Binary flag for URL shorteners
- ip_address: Binary flag if domain is IP address
- suspicious_tld: Binary flag for suspicious TLDs
- suspicious_keywords: Count of suspicious keywords
Domain-Based Features (8 features)
python# Domain Analysis
- domain_length: Length of domain name
- domain_age: Age of domain in days
- domain_entropy: Shannon entropy of domain
- subdomain_count: Number of subdomains
- has_https: HTTPS availability
- ssl_cert_valid: SSL certificate validity
- domain_reputation: Reputation score from threat feeds
- registrar_reputation: Registrar trustworthiness score
Content-Based Features (5 features)
python# Website Content Analysis
- title_similarity: Similarity to known brands
- favicon_similarity: Favicon comparison with legitimate sites
- form_count: Number of forms on page
- external_links: Number of external links
- javascript_suspicious: Suspicious JavaScript patterns
🤖 Machine Learning Models
Model Comparison
ModelAccuracyPrecisionRecallF1-ScoreTraining TimeRandom Forest96.2%95.8%96.6%96.2%45sGradient Boosting97.1%96.9%97.3%97.1%2m 15sXGBoost97.3%97.1%97.5%97.3%1m 30sNeural Network96.8%96.5%97.1%96.8%3m 45s
Best Model: XGBoost
python# Optimized hyperparameters
best_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}



project 4 : DNS Threat Detection: DGA & Tunneling Detection System
Problem Statement: The DNS Security Challenge
Domain Name System (DNS) protocols have become critical attack vectors for sophisticated cyber threats. Traditional security solutions struggle with two particularly evasive techniques:

Domain Generation Algorithms (DGAs)
Malware uses DGAs to generate thousands of random domain names daily, enabling:

Command-and-control (C2) communication

Evasion of domain blacklists

Resilient botnet operations
Detection Challenge: Differentiating algorithmically generated domains from legitimate human-created domains

DNS Tunneling
Attackers encode data in DNS queries to:

Exfiltrate sensitive data

Bypass network firewalls

Establish covert communication channels
Detection Challenge: Identifying malicious payloads in seemingly normal DNS traffic

The Critical Gap: Traditional signature-based detection fails against these threats due to:

Constantly changing domain patterns (DGAs)

Encryption and obfuscation techniques (tunneling)

Massive volume of DNS traffic (over 100B daily queries)

Low false-positive tolerance in enterprise environments


This project implements a multi-layered detection system combining:

Deep Learning for sequence pattern recognition (DGA)

Anomaly Detection for identifying outliers (tunneling)

Hybrid Rule Engine to reduce false positives

Key Features
Real-time DGA Detection: 98.3% accuracy on Cryptolocker variants

Tunneling Identification: 94.1% detection rate for iodine tunneling

Hybrid Analysis: 37% fewer false positives than pure ML approaches

Test Infrastructure: Custom DGA generator for model validation

Production-Ready API: <10ms/query processing latency

2. Model Specifications
DGA Detection (LSTM Network)

Character-level embeddings

Bidirectional LSTM layers

Attention mechanism

Output: Malicious probability (0-1)

Tunneling Detection (Isolation Forest)

Features:

Shannon entropy of domain

Domain length

Vowel-consonant ratio

Subdomain depth

Request frequency

Response size variability

Output: Anomaly score (-1 to 1)

Performance Benchmarks
Model	Precision	Recall	F1-Score	AUC	Throughput
DGA Detection	97.2%	96.8%	97.0%	0.992	850 qps
Tunneling Det.	94.5%	92.1%	93.3%	0.967	1200 qps
