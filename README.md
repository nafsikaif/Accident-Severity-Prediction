# 🚗 Accident Severity Prediction Using Machine Learning
### *An End-to-End, Leakage-Free Applied Machine Learning Pipeline & Interactive Decision Support System*

> **Target Audience:** Second-Year B.Tech in Artificial Intelligence  
> **Course / Project Type:** Minor Project in Applied Machine Learning & Data Engineering  
> **Repository Name:** `Accident_Severity_Prediction`

---

## 📌 Executive Summary

Road traffic accidents represent one of the leading causes of preventable mortality globally. In intelligent transportation systems and road safety engineering, predicting crash severity enables emergency responders, highway authorities, and urban planners to prioritize infrastructure interventions and allocate emergency medical resources effectively.

This project delivers an end-to-end, production-grade machine learning system that predicts accident severity into three standardized categories:
1. **Slight Collision** (Minor damage / slight injury)
2. **Serious Collision** (Hospitalization / significant vehicular damage)
3. **Fatal Collision** (Loss of human life)

The core engineering architecture strictly addresses real-world machine learning pitfalls:
- **Zero Target Leakage:** Drops all post-collision attributes (casualty statistics, hospital reports, investigation verdicts).
- **Severe Class Imbalance Mitigation:** Counteracts empirical road crash skewness (~80% Slight, ~17% Serious, ~3% Fatal) via inverse-frequency cost weighting.
- **Leakage-Free Preprocessing:** Bundles feature imputation and one-hot encoding strictly within `sklearn.pipeline.Pipeline` fitted only on training partitions.
- **Production-Ready Web GUI:** A two-page Streamlit dashboard designed with a white-and-lavender theme for live viva demonstrations and policy scenario analysis.

---

## 📂 System Architecture & Folder Structure

```
Accident_Severity_Prediction/
├── data/
│   └── accidents.csv            # Sourced from Kaggle: "Road Accident Severity in India" (or auto-generated fallback)
├── models/
│   └── best_model.joblib        # Fitted Scikit-learn Pipeline (Preprocessor + Winning Classifier)
├── reports/
│   ├── model_metrics.csv        # Test-set comparison metrics across models
│   └── confusion_matrices.joblib# Serialized test confusion matrices for UI rendering
├── src/
│   ├── __init__.py              # Package marker
│   ├── config.py                # Global paths, real column definitions, target mappings, random seeds
│   ├── data_loader.py           # Ingestion, validation, column verification, synthetic fallback generator
│   └── pipeline.py              # ColumnTransformer preprocessing pipeline builder
├── train_model.py               # Inspection, training, evaluation, model selection, and artifact serialization
├── app.py                       # Polished multi-page Streamlit GUI with custom white & lavender theme
├── requirements.txt             # Pinned minimal dependencies
└── README.md                    # Setup, execution, dataset sourcing, and viva preparation
```

---

## 🔬 Data Engineering & Anti-Leakage Protocol

### 1. Dataset Sourcing (Kaggle: "Road Accident Severity in India")
- **Raw File:** Often distributed on Kaggle as `Road.csv` or `RTA Dataset.csv`. Place this file at `data/accidents.csv`.
- **Dynamic Schema Inspection:** `src/data_loader.py` dynamically resolves column aliases (e.g., `Types_of_Junction` vs `Type_of_junction`) and maps target values to `Slight`, `Serious`, and `Fatal`.
- **Synthetic Fallback Generator:** If `data/accidents.csv` is missing, the system prints a diagnostic banner and synthesizes a realistic 2,000-row dataset conforming to empirical distribution statistics so tests run out-of-the-box.

### 2. Physical Pre-Crash Feature Whitelist
Predictions can only rely on data points that **physically existed prior to the collision**:
- **Driver Demographics:** `Age_band_of_driver`, `Sex_of_driver`, `Driving_experience`
- **Vehicular Category:** `Type_of_vehicle`
- **Kinematic Scale:** `Number_of_vehicles_involved`
- **Environmental & Road Conditions:** `Weather_conditions`, `Road_surface_conditions`, `Light_conditions`, `Road_surface_type`, `Type_of_junction`, `Day_of_week`

### 3. Strict Anti-Leakage Exclusion
The following variables are explicitly eliminated from the feature matrix because they are measured **after** the crash occurs:
| Variable | Leakage Category | Rationale for Exclusion |
| :--- | :--- | :--- |
| `Number_of_casualties` | Post-Crash Consequence | The number of casualties is a direct outcome of severe impact. |
| `Casualty_severity` | Target Derivative | Directly mirrors or leaks the target severity classification. |
| `Casualty_class` | Post-Crash Observation | Describes whether victims were pedestrians/passengers after the event. |
| `Sex_of_casualty` | Victim Attribute | Measured during post-crash emergency medical triage. |
| `Age_band_of_casualty` | Victim Attribute | Measured during post-crash emergency medical triage. |
| `Work_of_casuality` | Victim Attribute | Gathered from hospital admissions or post-incident forms. |
| `Fitness_of_casuality` | Medical Record | Evaluated during hospital treatment. |
| `Pedestrian_movement` | Investigation Finding | Recorded by police officers during reconstruction. |
| `Cause_of_accident` | Legal Determination | Assigned weeks post-crash following forensic investigation. |
| `Accident_injury_type` | Medical Diagnosis | Direct proxy for bodily trauma level. |

---

## ⚙️ Preprocessing & Modeling Methodology

### Preprocessing Architecture (`ColumnTransformer`)
1. **Numerical Features** (`Number_of_vehicles_involved`):
   - `SimpleImputer(strategy='median')`: Resilient against skewness and outliers.
   - `StandardScaler()`: Zero-centers mean and normalizes unit variance ($\mu=0, \sigma=1$).
2. **Categorical Features** (10 pre-crash discrete attributes):
   - `SimpleImputer(strategy='most_frequent')`: Imputes missing categorical records.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Creates binary indicator columns and gracefully absorbs unseen categories during live inference without runtime exceptions.

### Class Imbalance Mitigation
Empirical road data suffers from severe imbalance ($\approx 79\%$ Slight, $\approx 17\%$ Serious, $\approx 4\%$ Fatal). To prevent models from collapsing into majority-class classifiers:
- All candidate models implement `class_weight='balanced'`.
- Loss functions weight each sample inversely proportional to its class frequency:
  $$w_j = \frac{N}{K \cdot n_j}$$
  *(Where $N$ is total samples, $K$ is number of classes, and $n_j$ is the number of samples in class $j$)*.

### Candidate Algorithms Evaluated
1. **Logistic Regression** (`max_iter=1000`, `class_weight='balanced'`, `random_state=42`)
2. **Decision Tree Classifier** (`max_depth=6`, `class_weight='balanced'`, `random_state=42`)
3. **Random Forest Classifier** (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`)

### Model Selection Criterion
Standard accuracy is uninformative under high class imbalance. Model selection is governed strictly by the **Held-Out Macro F1-Score**:
$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$

---

## 📊 Benchmark Results (Test Set Evaluation)

*Evaluated on an 80/20 Stratified Test Partition ($N_{\text{test}} = 400$):*

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.7125** | **0.5535** | **0.5352** | **0.5535** | **0.5431** | 🏆 **Selected** |
| Decision Tree | 0.5200 | 0.5179 | 0.4048 | 0.5179 | 0.3934 | Baseline |
| Logistic Regression | 0.5150 | 0.4514 | 0.3870 | 0.4514 | 0.3594 | Baseline |

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ (Tested up to Python 3.13)
- Windows PowerShell, Command Prompt, or Linux/macOS terminal

### Step 1: Clone or Navigate to Project
```powershell
cd C:\Users\nafsi\.gemini\antigravity\scratch\Accident_Severity_Prediction
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Inspection, Training & Benchmark
```bash
python train_model.py
```
*Output Artifacts Produced:*
- `reports/model_metrics.csv`
- `reports/confusion_matrices.joblib`
- `models/best_model.joblib`

### Step 5: Launch the Streamlit GUI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎨 Streamlit Web Application Interface

The dashboard follows a clean white and soft lavender design system:
- **Page 1: Predict Severity**
  - Production Pipeline status indicator.
  - Interactive 2-column input form for pre-crash driver demographics and road attributes.
  - One-click inference button rendering high-contrast severity status badges:
    - `Slight`: Soft Green (`#C8E6C9`, text `#1B5E20`)
    - `Serious`: Soft Amber (`#FFE082`, text `#E65100`)
    - `Fatal`: Soft Red (`#FFCDD2`, text `#B71C1C`)
  - Class probability distribution horizontal bar chart.
  - Academic & empirical disclaimer.
- **Page 2: Model Benchmark & Analytics**
  - Performance comparison table highlighting the winning algorithm.
  - Interactive confusion matrix heatmap styled in a custom lavender gradient.
  - Pie chart visualizing the ground-truth severity imbalance.
  - Engineering audit trail documenting anti-leakage defenses.

---

## 🎓 Viva Voce & Academic Defense Guide

### Question 1: Why do casualty attributes (`Number_of_casualties`, `Casualty_severity`, etc.) represent target leakage in accident prediction?
**Answer:**
Target leakage (or data leakage) occurs when information that is not physically available at the time of prediction is introduced into the training feature set. 

In an operational deployment, an accident severity prediction model is intended to assist urban traffic management or route navigation algorithms *before* or *at the immediate moment* an accident occurs. Casualty attributes—such as `Number_of_casualties`, `Casualty_class`, `Work_of_casuality`, or `Casualty_severity`—are post-collision outcomes determined minutes to hours later during emergency hospital triage and police reporting. 

If casualty metrics are included as input features, the model learns a trivial mapping: *if casualties > 0 or casualty injury is severe, predict serious/fatal*. While this artificially yields near-perfect training accuracy ($\approx 99\%$), the model becomes completely unusable in real-world early-warning deployments where casualty counts are unknown prior to impact.

---

### Question 2: Why is the Macro F1-Score vastly superior to Standard Accuracy for this dataset?
**Answer:**
Standard Classification Accuracy measures the ratio of correct predictions over total predictions:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

In road accident datasets, the distribution of classes is heavily skewed:
- **Slight:** $\approx 80\%$
- **Serious:** $\approx 17\%$
- **Fatal:** $\approx 3\%$

A naive, degenerate classifier that predicts "Slight" for every single collision will achieve an accuracy of $\approx 80\%$, yet it possesses zero diagnostic utility because it misclassifies $100\%$ of life-threatening and fatal accidents (Recall for Fatal $= 0\%$).

The **Macro F1-Score** resolves this by computing the harmonic mean of Precision and Recall independently for each class and taking their unweighted arithmetic average:
$$\text{Macro F1} = \frac{\text{F1}_{\text{Slight}} + \text{F1}_{\text{Serious}} + \text{F1}_{\text{Fatal}}}{3}$$
Because all classes are treated with equal weight regardless of sample size, a failure to detect the minority `Fatal` class severely degrades the Macro F1-score. Thus, optimizing for Macro F1 guarantees that the model learns discriminatory boundaries for rare, high-consequence accidents.

---

### Question 3: How does `class_weight='balanced'` modify cost-function loss gradients mathematically?
**Answer:**
Under standard empirical risk minimization, each training example $(x_i, y_i)$ contributes equally to the loss function:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i; \theta), y_i)$$

When `class_weight='balanced'` is specified, a cost multiplier $w_{y_i}$ is applied to each observation based on the inverse class frequency:
$$w_j = \frac{N}{K \cdot n_j}$$
where $N$ is total training observations, $K$ is the number of classes (here, 3), and $n_j$ is the number of samples in class $j$.

The weighted loss function becomes:
$$\mathcal{L}_{\text{balanced}}(\theta) = \frac{1}{N} \sum_{i=1}^N w_{y_i} \cdot \ell(f(x_i; \theta), y_i)$$

During gradient descent, the gradient update vector with respect to weights $\theta$ is:
$$\nabla_\theta \mathcal{L}_{\text{balanced}}(\theta) = \frac{1}{N} \sum_{i=1}^N w_{y_i} \cdot \nabla_\theta \ell(f(x_i; \theta), y_i)$$

For the minority `Fatal` class ($n_{\text{Fatal}} \ll N$), $w_{\text{Fatal}}$ is significantly larger than $w_{\text{Slight}}$. Consequently, errors on fatal accident instances generate substantially larger gradient magnitudes, penalizing the model heavily for misclassifying minority instances and pushing the decision boundary toward protecting high-severity predictions.

---

### Question 4: Why is `handle_unknown='ignore'` mandatory when configuring `OneHotEncoder` for production?
**Answer:**
In multi-class categorical tabular problems, training data represents a finite sample of categorical levels. In live production or during cross-validation, incoming observations may present categories that were absent in the training split (e.g., a newly introduced vehicle classification, an unobserved weather combination, or unexpected typographical noise).

By default, scikit-learn's `OneHotEncoder(handle_unknown='error')` raises a `ValueError: Found unknown categories [...]` whenever an unseen string is encountered, immediately crashing the inference server or web GUI.

By setting `handle_unknown='ignore'`:
1. When an unknown category is encountered at test or inference time, the encoder assigns an all-zero vector ($[0, 0, \dots, 0]$) across the one-hot columns corresponding to that feature.
2. The pipeline continues execution without exception.
3. The model falls back to the intercept/prior weights or other available feature signals, ensuring high availability, robustness, and fault tolerance in production environments.

---

### Question 5: Compare Decision Tree splits versus Random Forest bagging in terms of variance, bias, and susceptibility to class imbalance.
**Answer:**

| Dimension | Single Decision Tree (CART) | Random Forest (Bagging Ensemble) |
| :--- | :--- | :--- |
| **Splitting Mechanism** | Evaluates **all** $p$ features at each node to pick the split maximizing Gini Impurity reduction or Information Gain. | Evaluates a **random subset** of features (typically $\sqrt{p}$) at each node split across $B$ independent bootstrapped trees. |
| **Variance & Bias** | **Low Bias, High Variance.** Individual deep trees easily overfit noisy training patterns, small changes in data cause large changes in tree structure. | **Low Bias, Substantially Reduced Variance.** Averaging decorrelated trees reduces variance by a factor related to ensemble size without sacrificing bias: $\text{Var}_{\text{RF}} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$. |
| **Imbalance Vulnerability** | **High.** Splitting criteria naturally favor the majority class because splits that isolate large homogeneous chunks of the majority class yield huge drops in overall impurity. | **Moderate / Controllable.** Combining bootstrap aggregating with `class_weight='balanced'` allows individual trees to vote collectively, mitigating false positive majority dominance. |
| **Decision Boundary** | Orthogonal step-wise partitions prone to jagged, irregular boundaries. | Smoothed ensemble probability contours yielding superior generalization on unseen test data. |

---

## 👨‍💻 Authors & Academic Attribution
- **Project Title:** Accident Severity Prediction Using Machine Learning
- **Institution:** Department of Artificial Intelligence & Machine Learning
- **Frameworks:** Python, Scikit-Learn, Pandas, NumPy, Streamlit, Seaborn, Matplotlib
