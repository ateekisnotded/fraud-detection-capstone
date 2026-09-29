"""Single-transaction inference using the saved Random Forest; never retrains."""

import joblib
import pandas as pd

from .preprocessing import MODEL_PATH, prepare_prediction_data


def load_prediction_artifact(model_path=MODEL_PATH):
    """Load the trusted model artifact used for prediction."""
    return joblib.load(model_path)


def predict_with_artifact(transaction, artifact):
    """Predict one transaction using an already-loaded model artifact."""
    X = prepare_prediction_data(pd.DataFrame([transaction]), artifact["preprocessing"])
    model = artifact["model"]
    fraud_index = list(model.classes_).index(1)
    return {
        "isFraud": int(model.predict(X)[0]),
        "fraud_probability": float(model.predict_proba(X)[0, fraud_index]),
    }


def predict_transaction(transaction, model_path=MODEL_PATH):
    """Load the artifact and predict one transaction.

    Required: step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest. Account IDs and isFraud are not needed.
    Only load model artifacts produced by this project or another trusted source.
    """
    return predict_with_artifact(transaction, load_prediction_artifact(model_path))
