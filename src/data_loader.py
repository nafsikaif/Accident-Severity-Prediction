"""Data ingestion, validation, schema verification, and synthetic fallback module.

Ensures strict anti-leakage feature selection, dynamic target column identification,
Kaggle schema compatibility, and automatic synthetic dataset generation when raw
data is missing.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    DATA_FILE_PATH,
    TARGET_CANDIDATES,
    TARGET_MAPPING,
    TARGET_CLASSES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    COLUMN_ALIASES,
    LEAKAGE_COLUMNS,
    FEATURE_CATEGORIES,
    SYNTHETIC_CLASS_DISTRIBUTION,
    RANDOM_STATE,
)


def generate_synthetic_dataset(
    output_path: Path = DATA_FILE_PATH,
    num_rows: int = 2000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generates a realistic synthetic road accident dataset matching the Kaggle schema.

    Preserves empirical road accident distributions (~80% Slight, ~17% Serious, ~3% Fatal)
    and includes both raw pre-crash features and post-crash attributes to test
    anti-leakage pruning pipelines realistically.

    Args:
        output_path: Path where the generated CSV should be saved.
        num_rows: Total number of observations to synthesize.
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing synthetic Kaggle-aligned accident data.
    """
    rng = np.random.default_rng(random_state)

    # 1. Synthesize target with severe class imbalance
    classes = list(SYNTHETIC_CLASS_DISTRIBUTION.keys())
    probs = list(SYNTHETIC_CLASS_DISTRIBUTION.values())
    targets = rng.choice(classes, size=num_rows, p=probs)

    # 2. Synthesize pre-crash categorical features
    data: Dict[str, Any] = {}
    for feature, categories in FEATURE_CATEGORIES.items():
        data[feature] = rng.choice(categories, size=num_rows)

    # 3. Synthesize pre-crash numerical features (Number_of_vehicles_involved)
    # Poisson-distributed with mean 2, clipped between 1 and 7
    vehicles = rng.poisson(lam=2.0, size=num_rows)
    vehicles = np.clip(vehicles, 1, 7)
    data["Number_of_vehicles_involved"] = vehicles

    # 4. Inject correlation: Fatal/Serious more likely on higher vehicle count or wet/dark roads
    for i in range(num_rows):
        if targets[i] == "Fatal":
            if rng.random() > 0.4:
                data["Light_conditions"][i] = "Darkness - no lighting"
                data["Road_surface_conditions"][i] = "Wet or damp"
                data["Number_of_vehicles_involved"][i] = max(data["Number_of_vehicles_involved"][i], 3)
        elif targets[i] == "Serious":
            if rng.random() > 0.5:
                data["Weather_conditions"][i] = rng.choice(["Raining", "Fog or mist"])

    # 5. Synthesize leakage & casualty columns (to prove leakage pruning works)
    casualties = []
    for t in targets:
        if t == "Fatal":
            casualties.append(int(rng.integers(1, 6)))
        elif t == "Serious":
            casualties.append(int(rng.integers(1, 4)))
        else:
            casualties.append(int(rng.choice([1, 2], p=[0.85, 0.15])))
    data["Number_of_casualties"] = casualties

    casualty_classes = ["Driver or rider", "Pedestrian", "Passenger"]
    data["Casualty_class"] = rng.choice(casualty_classes, size=num_rows)
    data["Casualty_severity"] = [f"{t} Injury" for t in targets]
    data["Sex_of_casualty"] = rng.choice(["Male", "Female"], size=num_rows, p=[0.7, 0.3])
    data["Age_band_of_casualty"] = rng.choice(["18-30", "31-50", "Over 51", "Under 18"], size=num_rows)
    data["Work_of_casuality"] = rng.choice(["Employee", "Self-employed", "Student", "Unemployed"], size=num_rows)
    data["Fitness_of_casuality"] = rng.choice(["Normal", "Blind", "Deaf"], size=num_rows, p=[0.96, 0.02, 0.02])
    data["Pedestrian_movement"] = rng.choice(["Not a Pedestrian", "Crossing road", "Walking on road"], size=num_rows)
    data["Cause_of_accident"] = rng.choice(["Moving Backward", "Overtaking", "Changing lane", "Driving carelessly"], size=num_rows)
    data["Accident_injury_type"] = targets

    # Set target column using Kaggle's typical naming
    data["Accident_severity"] = [f"{t} Injury" if t != "Fatal" else "Fatal injury" for t in targets]

    df = pd.DataFrame(data)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Generated synthetic Kaggle-aligned dataset: {output_path} ({num_rows} rows)")
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Dynamically locates the target severity column from candidates.

    Args:
        df: Input DataFrame.

    Returns:
        Exact column name of target.

    Raises:
        KeyError: If no candidate severity column is present.
    """
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    # Fallback case-insensitive check
    for col in df.columns:
        if "severity" in col.lower():
            return col
    raise KeyError(
        f"Could not locate an accident severity target column in: {df.columns.tolist()}. "
        f"Expected one of: {TARGET_CANDIDATES}"
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names using the canonical alias dictionary.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with normalized column names.
    """
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
        elif col.strip() in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col.strip()]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_and_validate_data(
    filepath: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Loads, inspects, validates, cleans, and isolates pre-crash features.

    If data file is missing, automatically triggers the synthetic fallback generator
    with a clear diagnostic banner.

    Args:
        filepath: Optional custom path to dataset CSV. Defaults to config DATA_FILE_PATH.
        verbose: Whether to print diagnostic schemas and validation summaries.

    Returns:
        Tuple of (X, y, metadata_dict)
            X: Clean pre-crash feature DataFrame (anti-leakage verified).
            y: Standardized 3-tier severity series ('Slight', 'Serious', 'Fatal').
            metadata_dict: Dictionary with dataset summary statistics.
    """
    path = Path(filepath) if filepath else DATA_FILE_PATH
    used_synthetic_fallback = False

    # Check if raw file exists; trigger synthetic fallback if absent
    if not path.exists():
        banner = (
            "\n" + "=" * 70 + "\n"
            "[WARNING] Kaggle dataset not found at:\n"
            f"   {path}\n"
            "   [Kaggle: 'Road Accident Severity in India' -> data/accidents.csv]\n"
            "   Automatically activating realistic synthetic fallback generator\n"
            "   (2,000 Kaggle-aligned records, ~80% Slight, ~17% Serious, ~3% Fatal).\n"
            + "=" * 70 + "\n"
        )
        print(banner)
        df_raw = generate_synthetic_dataset(output_path=path, num_rows=2000, random_state=RANDOM_STATE)
        used_synthetic_fallback = True
    else:
        df_raw = pd.read_csv(path)

    # Inspect schema
    if verbose:
        print("\n--- [DATASET SCHEMA INSPECTION] ---")
        print(f"Total Rows: {len(df_raw)}, Total Columns: {len(df_raw.columns)}")
        print(f"Columns List:\n{df_raw.columns.tolist()}\n")
        print("First 2 Sample Rows:")
        print(df_raw.head(2).to_string())
        print("-" * 50)

    # Standardize column naming aliases
    df = standardize_columns(df_raw.copy())

    # Detect dynamic target column
    target_col = detect_target_column(df)
    if verbose:
        print(f"[INFO] Target column detected: '{target_col}'")

    # Map target strings to canonical labels ('Slight', 'Serious', 'Fatal')
    y_raw = df[target_col].astype(str).str.strip()
    y_mapped = y_raw.map(TARGET_MAPPING)

    # Drop unmapped or null target instances if any exist
    valid_mask = y_mapped.notna() & y_mapped.isin(TARGET_CLASSES)
    if not valid_mask.all():
        dropped_count = int((~valid_mask).sum())
        if verbose:
            print(f"[WARN] Dropping {dropped_count} rows with unmapped target severity.")
        df = df[valid_mask].copy()
        y_mapped = y_mapped[valid_mask]

    y = y_mapped.astype("category")

    # Anti-leakage pruning: Explicitly drop post-crash & casualty attributes
    dropped_leakage: List[str] = []
    for leak_col in LEAKAGE_COLUMNS:
        if leak_col in df.columns:
            dropped_leakage.append(leak_col)

    if verbose and dropped_leakage:
        print("\n--- [STRICT ANTI-LEAKAGE PRUNING] ---")
        print(f"Eliminated {len(dropped_leakage)} post-crash / casualty leakage columns:")
        for col in dropped_leakage:
            print(f"  [-] EXCLUDED: {col}")
        print("-" * 50)

    # Whitelist physical pre-crash features present in dataframe
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [c for c in NUMERICAL_FEATURES if c in df.columns]

    selected_features = available_cat + available_num
    if not selected_features:
        raise ValueError(
            f"No verified pre-crash features could be extracted from: {df.columns.tolist()}"
        )

    X = df[selected_features].copy()

    # Ensure numeric columns are properly cast
    for num_col in available_num:
        X[num_col] = pd.to_numeric(X[num_col], errors="coerce")

    # Ensure categorical columns are strings
    for cat_col in available_cat:
        X[cat_col] = X[cat_col].astype(str)

    # Calculate class distribution
    class_counts = y.value_counts().to_dict()
    class_proportions = (y.value_counts(normalize=True) * 100).round(2).to_dict()

    if verbose:
        print("\n--- [CLASS DISTRIBUTION] ---")
        for cls, count in class_counts.items():
            prop = class_proportions.get(cls, 0.0)
            print(f"  Class '{cls}': {count:,} samples ({prop}%)")
        print("-" * 50 + "\n")

    metadata: Dict[str, Any] = {
        "total_samples": len(df),
        "target_column": target_col,
        "categorical_features": available_cat,
        "numerical_features": available_num,
        "selected_features": selected_features,
        "dropped_leakage": dropped_leakage,
        "class_counts": class_counts,
        "class_proportions": class_proportions,
        "used_synthetic_fallback": used_synthetic_fallback,
    }

    return X, y, metadata


if __name__ == "__main__":
    X, y, meta = load_and_validate_data()
    print("Self-test completed successfully.")
    print("Features matrix shape:", X.shape)
    print("Target distribution:", meta["class_counts"])
