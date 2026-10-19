"""Model training, evaluation, benchmark logging, and artifact serialization.

Executes stratified train-test splitting, fits three candidate algorithms
with class_weight='balanced', computes multi-class academic evaluation metrics,
selects the model with the highest Macro F1-score, and serializes the winning
pipeline and artifacts.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    MODEL_FILE_PATH,
    METRICS_FILE_PATH,
    CONFUSION_MATRICES_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_CLASSES,
)
from src.data_loader import load_and_validate_data
from src.pipeline import build_full_pipeline


def train_and_evaluate() -> Dict[str, Any]:
    """Orchestrates data ingestion, anti-leakage preparation, model training,

    evaluation across candidate algorithms, and winning model serialization.

    Returns:
        Dictionary containing benchmark summary, winning model name, and metrics.
    """
    print("\n" + "=" * 70)
    print("      ACCIDENT SEVERITY PREDICTION - ML TRAINING & BENCHMARK")
    print("=" * 70)

    # 1. Ingestion, Schema Inspection, and Anti-Leakage Filtering
    X, y, meta = load_and_validate_data(verbose=True)

    # 2. Stratified Train-Test Split (80/20)
    print(f"\n[INFO] Executing Stratified Train-Test Split (Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, random_state={RANDOM_STATE})")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"  Training Set Size:   {len(X_train):,} samples")
    print(f"  Held-out Test Size:  {len(X_test):,} samples")

    # 3. Define Candidate Classifiers with Balanced Class Weights
    # class_weight='balanced' inversely penalizes majority classes to ensure
    # fair attention to Fatal (minority ~3%) and Serious (~17%) collisions.
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    metrics_records = []
    fitted_pipelines = {}
    stored_confusion_matrices = {}

    cat_cols = meta["categorical_features"]
    num_cols = meta["numerical_features"]

    print("\n" + "=" * 70)
    print("              MODEL BENCHMARKING (TEST SET)")
    print("=" * 70)

    for name, clf in models.items():
        print(f"\n[TRAINING] {name} ...")
        # Build end-to-end pipeline (Preprocessor + Classifier)
        pipeline = build_full_pipeline(
            classifier=clf,
            categorical_cols=cat_cols,
            numerical_cols=num_cols,
        )

        # Fit strictly on X_train to prevent leakage
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on held-out test set
        y_pred = pipeline.predict(X_test)

        # Compute academic multi-class metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Compute confusion matrix with standard class ordering
        cm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES)
        stored_confusion_matrices[name] = {
            "matrix": cm,
            "classes": TARGET_CLASSES,
        }

        record = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Balanced Accuracy": round(bal_acc, 4),
            "Macro Precision": round(prec_macro, 4),
            "Macro Recall": round(rec_macro, 4),
            "Macro F1-Score": round(f1_macro, 4),
        }
        metrics_records.append(record)

        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  Macro Precision:   {prec_macro:.4f}")
        print(f"  Macro Recall:      {rec_macro:.4f}")
        print(f"  Macro F1-Score:    {f1_macro:.4f}")

        # Detailed per-class classification report
        print(f"\n  Classification Report for {name}:")
        report = classification_report(y_test, y_pred, labels=TARGET_CLASSES, zero_division=0)
        for line in report.strip().split("\n"):
            print(f"    {line}")

    # 4. Tabulate Benchmark Results
    df_metrics = pd.DataFrame(metrics_records)
    # Sort descending by Macro F1-Score
    df_metrics = df_metrics.sort_values(by="Macro F1-Score", ascending=False).reset_index(drop=True)

    # 5. Model Selection based strictly on Highest Macro F1-Score
    best_row = df_metrics.iloc[0]
    best_model_name = best_row["Model"]
    best_f1 = best_row["Macro F1-Score"]
    best_pipeline = fitted_pipelines[best_model_name]

    print("\n" + "=" * 70)
    print("              FINAL MODEL SELECTION RESULTS")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("-" * 70)
    print(f"[WINNER] Selected Model: {best_model_name}")
    print(f"   Held-Out Macro F1-Score: {best_f1:.4f}")
    print("=" * 70 + "\n")

    # 6. Save Artifacts for UI & Reproducibility
    # Save Metrics CSV
    METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(METRICS_FILE_PATH, index=False)
    print(f"[SAVED] Test Metrics: {METRICS_FILE_PATH}")

    # Save Confusion Matrices Joblib
    CONFUSION_MATRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stored_confusion_matrices, CONFUSION_MATRICES_PATH)
    print(f"[SAVED] Confusion Matrices: {CONFUSION_MATRICES_PATH}")

    # Save Best Model Pipeline Joblib
    MODEL_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_pipeline,
            "classes": TARGET_CLASSES,
            "feature_names": meta["selected_features"],
            "categorical_features": meta["categorical_features"],
            "numerical_features": meta["numerical_features"],
            "macro_f1": best_f1,
        },
        MODEL_FILE_PATH,
    )
    print(f"[SAVED] Winning Model Pipeline: {MODEL_FILE_PATH}")

    return {
        "best_model_name": best_model_name,
        "best_macro_f1": best_f1,
        "metrics_df": df_metrics,
    }


if __name__ == "__main__":
    train_and_evaluate()
