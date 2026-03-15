from __future__ import annotations
import pandas as pd

from src.models import get_model
from src.evaluate import evaluate_classifier


def run_model_comparison(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_names: list[str],
):
    results_rows = []
    fitted_models = {}
    detailed_results = {}

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    for model_name in model_names:
        model = get_model(model_name)
        model.fit(X_train, y_train)

        val_metrics = evaluate_classifier(model, X_val, y_val)
        test_metrics = evaluate_classifier(model, X_test, y_test)

        results_rows.append({
            "model": model_name,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_avg_precision": val_metrics["avg_precision"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_avg_precision": test_metrics["avg_precision"],
        })

        fitted_models[model_name] = model
        detailed_results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

    results_df = pd.DataFrame(results_rows).sort_values(
        by="test_roc_auc", ascending=False
    ).reset_index(drop=True)

    return results_df, fitted_models, detailed_results