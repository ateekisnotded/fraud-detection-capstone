"""Notebook-equivalent feature preparation for training and prediction."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Fraud_Analysis_Dataset.xlsx"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
RAW_FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "type",
]
FEATURES = RAW_FEATURES[:-1] + [
    "orig_balance_change", "dest_balance_change",
    "orig_balance_error", "dest_balance_error", "type",
]


def load_dataset(path=DATASET_PATH):
    """Resolve the default dataset relative to this project, not the working directory."""
    return pd.read_excel(path)


def engineer_features(transactions):
    """Copy inputs and apply the four original notebook formulas."""
    missing = set(RAW_FEATURES) - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing transaction fields: {sorted(missing)}")
    frame = transactions.copy()
    frame["orig_balance_change"] = frame["oldbalanceOrg"] - frame["newbalanceOrig"]
    frame["dest_balance_change"] = frame["newbalanceDest"] - frame["oldbalanceDest"]
    frame["orig_balance_error"] = frame["amount"] - frame["orig_balance_change"]
    frame["dest_balance_error"] = frame["amount"] - frame["dest_balance_change"]
    return frame[FEATURES].copy()


def prepare_training_data(frame):
    """Encode before splitting, exactly as in the notebook; return X, y, schema."""
    features = engineer_features(frame)
    X = pd.get_dummies(features, columns=["type"], drop_first=True)
    y = frame["isFraud"].copy()
    schema = {
        "type_categories": sorted(features["type"].dropna().unique().tolist()),
        "feature_columns": X.columns.tolist(),
        "feature_dtypes": {name: str(dtype) for name, dtype in X.dtypes.items()},
        "raw_features": RAW_FEATURES.copy(),
        "encoding": "pandas.get_dummies",
        "drop_first": True,
    }
    return X, y, schema


def prepare_prediction_data(transactions, schema):
    """Use training categories so a single row gets the same dummy columns."""
    features = engineer_features(transactions)
    valid = features["type"].isin(schema["type_categories"])
    if not valid.all():
        raise ValueError("Transaction type must be one of " + str(schema["type_categories"]))
    features["type"] = pd.Categorical(
        features["type"], categories=schema["type_categories"]
    )
    X = pd.get_dummies(features, columns=["type"], drop_first=schema["drop_first"])
    # Preserve numeric input precision; dummy dtypes match the saved training schema.
    X = X.reindex(columns=schema["feature_columns"])
    for name in X.columns:
        if name.startswith("type_"):
            X[name] = X[name].astype(schema["feature_dtypes"][name])
    return X
