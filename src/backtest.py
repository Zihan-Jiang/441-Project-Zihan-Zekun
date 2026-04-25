from __future__ import annotations
import pandas as pd
from sklearn.base import clone

from src.evaluate import evaluate_classifier


def rolling_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model,
    train_window: int = 756,   # Approximately 3 years of trading days
    test_window: int = 63      # Approximately one quarter
) -> pd.DataFrame:
    """
    Very simple rolling backtest:
    - Use fixed-length train window
    - Predict next test_window block
    """
    results = []

    start = 0
    n = len(df)

    while start + train_window + test_window <= n:
        train_df = df.iloc[start:start + train_window].copy()
        test_df = df.iloc[start + train_window:start + train_window + test_window].copy()

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        metrics = evaluate_classifier(fitted_model, X_test, y_test)

        results.append({
            "train_start": train_df["Date"].min(),
            "train_end": train_df["Date"].max(),
            "test_start": test_df["Date"].min(),
            "test_end": test_df["Date"].max(),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

        start += test_window

    return pd.DataFrame(results)