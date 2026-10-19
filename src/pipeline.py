"""Preprocessing and full pipeline builder module.

Constructs an anti-leakage ColumnTransformer that handles numerical imputation/scaling
and categorical imputation/one-hot encoding with handle_unknown='ignore'.
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for pre-crash accident features.

    Pipeline Design:
      - Numerical Pipeline:
          1. SimpleImputer(strategy='median'): Robust to right-skewed vehicle counts.
          2. StandardScaler(): Normalizes numerical variance for gradient & linear models.
      - Categorical Pipeline:
          1. SimpleImputer(strategy='most_frequent'): Handles missing field recordings.
          2. OneHotEncoder(handle_unknown='ignore', sparse_output=False): Prevents runtime
             exceptions when encountering novel category strings during inference.

    Args:
        categorical_cols: List of categorical feature names. Defaults to config list.
        numerical_cols: List of numerical feature names. Defaults to config list.

    Returns:
        Configured ColumnTransformer instance.
    """
    cats = categorical_cols if categorical_cols is not None else CATEGORICAL_FEATURES
    nums = numerical_cols if numerical_cols is not None else NUMERICAL_FEATURES

    transformers = []

    if nums:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numerical_pipeline, nums))

    if cats:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipeline, cats))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_pipeline(
    classifier,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Pipeline:
    """Combines the preprocessor ColumnTransformer and a classifier into an end-to-end Pipeline.

    By encapsulating feature transformation and classification in an atomic Pipeline,
    we ensure strict prevention of data leakage: preprocessor statistics (e.g., median,
    mean, std, category vocabularies) are fitted strictly on X_train.

    Args:
        classifier: An instantiated scikit-learn compatible classifier.
        categorical_cols: List of categorical feature names.
        numerical_cols: List of numerical feature names.

    Returns:
        End-to-end Pipeline(steps=[('preprocessor', ...), ('classifier', ...)])
    """
    preprocessor = build_preprocessor(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return full_pipeline
