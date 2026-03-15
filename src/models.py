from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_model(model_name: str):
    name = model_name.lower()

    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ])

    if name == "svm_linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", probability=True, random_state=42))
        ])

    if name == "svm_rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42))
        ])

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    raise ValueError(f"Unsupported model_name: {model_name}")