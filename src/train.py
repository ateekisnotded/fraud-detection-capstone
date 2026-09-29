"""Run with python -B -m src.train from the project root."""

import platform
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .preprocessing import MODEL_PATH, load_dataset, prepare_training_data

NOTEBOOK_METRICS = {
    "Logistic Regression": [1.0, 0.8991228070175439, 0.9468822170900693, 0.9911535460340006],
    "Random Forest": [1.0, 0.9956140350877193, 0.9978021978021978, 0.9999572582129987],
    "Gradient Boosting": [1.0, 0.9956140350877193, 0.9978021978021978, 0.999677792682606],
}
METRIC_NAMES = ["Precision", "Recall", "F1", "ROC-AUC"]


def train():
    X, y, schema = prepare_training_data(load_dataset())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }
    print(f"Features: {X.shape[1]} | Training: {len(X_train)} | Testing: {len(X_test)}")
    rows, evaluations, discrepancies = [], {}, []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        scores = [
            precision_score(y_test, predicted), recall_score(y_test, predicted),
            f1_score(y_test, predicted), roc_auc_score(y_test, probability),
        ]
        rows.append({"Model": name, **dict(zip(METRIC_NAMES, scores))})
        evaluations[name] = {
            "metrics": dict(zip(METRIC_NAMES, scores)),
            "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
            "classification_report": classification_report(y_test, predicted, output_dict=True),
        }
        print(f"\n{name}\n{classification_report(y_test, predicted)}")
        print("Confusion matrix (actual rows, predicted columns; labels 0, 1):")
        print(confusion_matrix(y_test, predicted))
        for metric, actual, expected in zip(METRIC_NAMES, scores, NOTEBOOK_METRICS[name]):
            if abs(actual - expected) > 1e-6:
                discrepancies.append(f"{name} {metric}: {actual:.15f} vs notebook {expected:.15f}")

    comparison = pd.DataFrame(rows)
    print("\nModel comparison")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.15f}"))
    if discrepancies:
        raise RuntimeError(
            "Notebook reproduction differs by more than 0.000001; model was not saved.\n"
            + "\n".join(discrepancies)
        )

    artifact = {
        "model": models["Random Forest"],
        "preprocessing": schema,
        "evaluations": evaluations,
        "model_parameters": {name: model.get_params() for name, model in models.items()},
        "split": {"test_size": 0.20, "random_state": 42, "stratify": "isFraud"},
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in
               ["pandas", "numpy", "scipy", "scikit-learn", "openpyxl", "joblib"]},
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nNotebook metrics reproduced within 0.000001. Saved: {MODEL_PATH}")
    return comparison


if __name__ == "__main__":
    train()
