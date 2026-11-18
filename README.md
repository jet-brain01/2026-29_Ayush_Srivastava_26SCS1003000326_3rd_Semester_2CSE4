
<div align="center">

# Machine Learning-Based Health Risk Prediction System

**An Artificial Intelligence & Machine Learning Internship Project**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/docs/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge)]()

*An end-to-end supervised machine learning pipeline and interactive web dashboard prototype designed to analyze clinical health parameters and classify cardiovascular health-risk categories.*

</div>

---

> [!WARNING]
> **Medical Disclaimer:** This project is an educational prototype developed during an academic AI/ML internship. It is trained on a small historical dataset (303 records) and has not undergone clinical validation. Its outputs are strictly illustrative and must not be used for medical diagnosis, treatment, or clinical decision-making.

---

## Table of Contents

- [Internship & Academic Credentials](#-internship--academic-credentials)
- [Project Overview](#-project-overview)
- [Problem Statement & Objectives](#-problem-statement--objectives)
- [System Architecture](#-system-architecture)
- [Dataset Description](#-dataset-description)
- [Data Preprocessing & EDA](#-data-preprocessing--eda)
- [Machine Learning Models](#-machine-learning-models)
- [Execution Pipeline & Evaluation Framework](#-execution-pipeline--evaluation-framework)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Challenges & Limitations](#-challenges--limitations)
- [Future Scope](#-future-scope)
- [Documentation & Credentials](#-documentation--credentials)
- [References](#-references)
- [Author](#-author)

---

## Internship & Academic Credentials

| Attribute | Details |
| :--- | :--- |
| **Student Name** | Ayush Srivastava |
| **Roll Number** | `26SCS1003000326` |
| **Program** | B.Tech – Computer Science & Engineering (AI & ML) |
| **Academic Details** | 2nd Year \| 3rd Semester \| Section `2CSE4` |
| **University** | IILM University, Greater Noida |
| **Organization** | TechNova Solutions (TNS Solutions Pvt. Ltd.) |
| **Internship Domain** | Artificial Intelligence & Machine Learning |
| **Mode & Duration** | Virtual \| 4 Weeks (`18 June 2026` – `16 July 2026`) |
| **Certificate ID** | `TN-AI-ML-2026-0784` (Issued: `18 July 2026`) |

---

## Project Overview

The **Machine Learning-Based Health Risk Prediction System** demonstrates how supervised machine learning algorithms can process structured clinical parameters, identify hidden multi-variable patterns, and estimate preliminary health-risk categories.

Built over a 4-week virtual internship at **TechNova Solutions**, this project covers the complete applied ML lifecycle—from raw healthcare dataset exploration and preprocessing to comparative model training, clinical-metric evaluation, and web-based risk visualization using Flask.

---

## Problem Statement & Objectives

### The Problem

Manual analysis of multi-parameter clinical records (such as blood pressure, cholesterol, ECG readings, and heart rate) is complex and prone to oversight when evaluating non-linear interactions across multiple health indicators.

### Proposed Solution

An automated, modular supervised machine learning pipeline paired with a Flask-powered web dashboard that validates inputs, preprocesses features, executes inference, and presents intuitive risk categorizations.

### Key Objectives

1. **Understand** structured clinical healthcare datasets and domain attributes.
2. **Preprocess** raw data by handling missing values, encoding categorical variables, and scaling numeric features.
3. **Explore** feature distributions, correlations, and target class balance through EDA.
4. **Train** and benchmark multiple supervised ML classification algorithms.
5. **Evaluate** model reliability using healthcare-appropriate metrics (Recall, F1-Score, ROC-AUC, and Confusion Matrix).
6. **Generate** clear preliminary health-risk predictions from user-provided parameters.
7. **Visualize** risk outputs through a clean, accessible web dashboard interface.

---

## System Architecture

The system follows a decoupled 8-stage modular pipeline separating the user interface, validation layer, ML inference engine, and output visualization for maintainability.

### System Architecture Diagram

```mermaid
flowchart LR
    A[1. User] --> B[2. Health Input]
    B --> C[3. Validation]
    C --> D[4. Preprocessing]
    D --> E[5. Feature Selection]
    E --> F[6. Trained ML Model]
    F --> G[7. Risk Prediction]
    G --> H[8. Results Dashboard]
```

### Architecture Components

| Stage | Component | Responsibility |
| :---: | :--- | :--- |
| **1** | **User** | Initiates a health-risk check via the web interface |
| **2** | **Health Input** | Captures clinical parameters (vitals, demographics, ECG metrics) |
| **3** | **Validation** | Enforces data types and valid physiological value ranges |
| **4** | **Preprocessing** | Encodes categorical attributes and applies feature scaling |
| **5** | **Feature Selection** | Retains high-signal predictors and prunes noisy inputs |
| **6** | **Trained ML Model** | Executes classification inference via Scikit-learn |
| **7** | **Risk Prediction** | Assigns preliminary health-risk category (e.g., Low / Elevated) |
| **8** | **Results Dashboard** | Renders visual risk indicators and disclaimers |

---

## Dataset Description

This project utilizes the **UCI Heart Disease (Cleveland Clinic)** dataset from the UCI Machine Learning Repository.

- **Records:** 303 patient entries
- **Attributes:** 14 standard clinical features (13 predictors + 1 target)
- **Target Variable (`num`):** `0` (Heart disease absent) vs. `1–4` (Heart disease present, binarized to `0 / 1`)
- **Missing Values:** 6 rows total (`4` in `ca`, `2` in `thal`)

### Feature Classification & Key Attributes

| Category | Attributes |
| :--- | :--- |
| **Demographic** | `age`, `sex` |
| **Vitals & Lab** | `trestbps`, `chol`, `fbs` |
| **Cardiac & ECG** | `cp`, `thalach`, `restecg`, `exang` |
| **Diagnostic & Target** | `oldpeak`, `slope`, `ca`, `thal`, `num` |

### Dataset Attribute Description

| Attribute | Description | Data Type |
| :--- | :--- | :--- |
| `age` | Patient age in years | Integer |
| `trestbps` | Resting blood pressure (mm Hg) | Integer |
| `chol` | Serum cholesterol (mg/dl) | Integer |
| `thalach` | Maximum heart rate achieved | Integer |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| `cp` | Chest pain type (4 categories) | Categorical |
| `oldpeak` / `ca` / `thal` | ST depression, major vessels colored, thalassemia | Numeric / Categorical |
| `num` *(Target)* | Heart disease presence indicator | Binary (`0` / `1`) |

---

## Data Preprocessing & EDA

Raw clinical data is transformed into a model-ready format through the following pipeline:

```text
Raw Dataset
     |
     v
Data Cleaning
     |
     v
Missing Values Handling
     |
     v
Categorical Encoding
     |
     v
Feature Scaling
     |
     v
Processed Dataset
     |
     v
Exploratory Data Analysis (EDA)
```

### Preprocessing Techniques

- **Data Cleaning:** Removed duplicate records and handled the 6 missing entries in `ca` and `thal`.
- **Categorical Encoding:** Applied One-Hot Encoding to nominal clinical fields such as chest pain type (`cp`).
- **Feature Scaling:** Standardized numeric distributions using `StandardScaler` to ensure optimal convergence for distance- and margin-based algorithms (SVM, KNN).
- **Train-Test Split:** Partitioned the dataset into **80% Training** and **20% Testing** subsets.
- **Exploratory Data Analysis (EDA):** Analyzed feature distributions, class balance, and target correlation—highlighting strong predictive relevance in **Chest Pain (`cp`)**, **Max Heart Rate (`thalach`)**, **ST Depression (`oldpeak`)**, **Age**, and **Resting Blood Pressure (`trestbps`)**.

---

## Machine Learning Models

Five supervised classification algorithms were analyzed and compared to balance interpretability, variance, and predictive accuracy.

| Algorithm | Working Principle | Advantage | Limitation | Project Fit |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Sigmoid probability boundary | Fast, highly interpretable | Assumes linear relationships | Strong baseline |
| **Decision Tree** | Rule-based recursive splits | No feature scaling required | Prone to overfitting | Visual decision rules |
| **Random Forest** | Ensemble voting across trees | Robust, low variance | Less interpretable | High accuracy (Primary) |
| **Support Vector Machine (SVM)** | Maximum-margin hyperplane | Effective in high dimensions | Sensitive to feature scaling | Small datasets |
| **K-Nearest Neighbors (KNN)** | Distance-based neighbor vote | Simple, non-parametric | Sensitive to noisy data | Benchmarking |

### Core Python Pipeline Snippet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and clean dataset
df = pd.read_csv("heart_cleveland.csv").dropna()
X, y = df.drop("target", axis=1), df["target"]

# Train-test split (80/20)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)

# Train Random Forest model and run inference
model = RandomForestClassifier(random_state=42).fit(X_tr_s, y_tr)
risk_pred = model.predict(scaler.transform(X_te[:1]))
```

---

## Execution Pipeline & Evaluation Framework

### Training & Deployment Gate

1. **Phase 1 — Training:** Load processed dataset → Split Train/Test → Select ML Algorithm → Train Model.
2. **Phase 2 — Evaluation Gate:** Evaluate Accuracy, Recall, and F1-Score. If performance is below threshold, tune hyperparameters and retrain; otherwise, serialize and save the trained model.
3. **Phase 3 — Prediction:** Accept new clinical input → Generate Risk Prediction → Display on Dashboard.

### Why Accuracy Alone Is Insufficient in Healthcare

In medical risk screening, **False Negatives (FN)** carry the highest clinical cost because an at-risk individual is misclassified as healthy. Therefore, evaluation prioritizes **Recall (Sensitivity)**, **F1-Score**, and **ROC-AUC** alongside Accuracy.

**Accuracy:**

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Precision:**

$$
Precision = \frac{TP}{TP + FP}
$$

**Recall (Sensitivity):**

$$
Recall = \frac{TP}{TP + FN}
$$

**F1-Score:**

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### Evaluation Metrics

| Metric | Clinical Interpretation |
| :--- | :--- |
| **True Positive (TP) / True Negative (TN)** | Correctly identified at-risk and healthy cases |
| **False Positive (FP — Type I Error)** | Healthy case flagged as at-risk |
| **False Negative (FN — Type II Error)** | At-risk case missed *(highest clinical cost)* |
| **ROC-AUC** | Overall class separation capability across decision thresholds |

> [!NOTE]
> **Experimental Metrics:** Update the table below with your verified local test-run outputs once finalized. Do not use estimated or fabricated performance values.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |
| Logistic Regression | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |
| SVM | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Programming Language** | Python 3.x |
| **Data Processing & ML** | NumPy, Pandas, Scikit-learn |
| **Data Visualization** | Matplotlib, Seaborn |
| **Web Backend & API** | Flask |
| **Frontend UI** | HTML5, CSS3, JavaScript |
| **Database (Optional)** | SQLite |

---

## Repository Structure

```text
2026-29_Ayush_Srivastava_26SCS1003000326_3rd_Semester_2CSE4/
│
├── data/
│   └── heart_cleveland.csv
│
├── docs/
│   ├── Internship_Certificate_Ayush_Srivastava_26SCS1003000326.pdf
│   ├── Internship_Offer_Letter_Ayush_Srivastava_26SCS1003000326.pdf
│   └── Internship_PPT_Ayush_Srivastava_26SCS1003000326.pptx
│
├── notebooks/
│   └── eda_and_model_training.ipynb
│
├── src/
│   ├── app.py
│   ├── train.py
│   └── templates/
│
├── requirements.txt
│
└── README.md
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jet-brain01/2026-29_Ayush_Srivastava_26SCS1003000326_3rd_Semester_2CSE4.git

cd 2026-29_Ayush_Srivastava_26SCS1003000326_3rd_Semester_2CSE4
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**On Windows:**

```bash
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Core packages:

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `flask`

### 4. Run the Application

```bash
python src/app.py
```

Open your browser and navigate to:

`http://127.0.0.1:5000/`

Interact with the Health Risk Prediction Dashboard.

---

## Challenges & Limitations

### Key Challenges

- Handling incomplete or noisy clinical records (`ca` and `thal` missing entries).
- Selecting optimal features while preventing overfitting on a compact dataset.
- Balancing **Recall vs. Precision** to minimize False Negatives.
- Accounting for patient data privacy and ethical AI considerations.

### System Limitations

- **Small Sample Size:** Limited to 303 records from the Cleveland Clinic subset.
- **Demographic Bias:** Potential lack of external generalization across diverse global populations.
- **No Clinical Validation:** Not validated through clinical trials; strictly for educational and research exploration.

---

## Future Scope

- **Larger, Diverse Datasets:** Train and validate on multi-center, demographically diverse clinical cohorts.
- **Explainable AI (XAI):** Integrate **SHAP** and **LIME** visualizations so predictions are transparent and interpretable for each individual input.
- **Advanced Modeling:** Explore gradient boosting (XGBoost, LightGBM) and deep learning architectures.
- **Privacy-Preserving Integration:** Implement secure authentication, encrypted storage, and federated learning concepts.
- **Clinical Collaboration:** Partner with domain healthcare professionals to refine feature engineering and risk thresholds.

---

## Documentation & Credentials

All internship documents and project presentation are included in this repository for academic reference.

| Document | File |
| :--- | :--- |
| **Internship Completion Certificate** | [View Certificate (PDF)](docs/Internship_Certificate_Ayush_Srivastava_26SCS1003000326.pdf) |
| **Internship Offer Letter** | [View Offer Letter (PDF)](docs/Internship_Offer_Letter_Ayush_Srivastava_26SCS1003000326.pdf) |
| **Project Presentation** | [View Presentation (PPTX)](docs/Internship_PPT_Ayush_Srivastava_26SCS1003000326.pptx) |

---

## References

- **UCI Machine Learning Repository:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease Dataset* — [archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Scikit-learn Documentation:** [scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **Python Official Documentation:** [docs.python.org/3/](https://docs.python.org/3/)
- **Pandas Documentation:** [pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **NumPy Documentation:** [numpy.org/doc/](https://numpy.org/doc/)
- **Flask Documentation:** [flask.palletsprojects.com/](https://flask.palletsprojects.com/)

---

## Author

**Ayush Srivastava**

B.Tech CSE (Artificial Intelligence & Machine Learning)  
IILM University, Greater Noida

**AI & ML Intern — TechNova Solutions**  
`June 2026 – July 2026`

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat-square&logo=github)](https://github.com/jet-brain01)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/<your-linkedin>)

---

<div align="center">

<sub>If you found this project useful for learning healthcare ML workflows, feel free to star the repository!</sub>

</div>
