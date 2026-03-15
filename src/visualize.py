from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_roc_curves(detailed_results: dict, split_name: str, save_path: str) -> None:
    plt.figure(figsize=(8, 6))

    for model_name, result_dict in detailed_results.items():
        metrics = result_dict[split_name]
        if metrics["fpr"] is not None and metrics["tpr"] is not None:
            auc_value = metrics["roc_auc"]
            plt.plot(metrics["fpr"], metrics["tpr"], label=f"{model_name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({split_name.title()})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curves(detailed_results: dict, split_name: str, save_path: str) -> None:
    plt.figure(figsize=(8, 6))

    for model_name, result_dict in detailed_results.items():
        metrics = result_dict[split_name]
        if metrics["pr_recall_curve"] is not None and metrics["pr_precision_curve"] is not None:
            ap_value = metrics["avg_precision"]
            plt.plot(
                metrics["pr_recall_curve"],
                metrics["pr_precision_curve"],
                label=f"{model_name} (AP={ap_value:.3f})"
            )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves ({split_name.title()})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix_for_model(
    detailed_results: dict,
    model_name: str,
    split_name: str,
    save_path: str
) -> None:
    metrics = detailed_results[model_name][split_name]
    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name} ({split_name.title()})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def extract_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Support:
    - Logistic Regression coefficients
    - Linear SVM coefficients
    - Random Forest feature_importances_
    - Gradient Boosting feature_importances_
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["model"]

    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        importance_type = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "importance"
    else:
        return pd.DataFrame(columns=["feature", "value", "abs_value", "type"])

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "value": values,
    })
    df_imp["abs_value"] = df_imp["value"].abs()
    df_imp["type"] = importance_type
    df_imp = df_imp.sort_values("abs_value", ascending=False).reset_index(drop=True)
    return df_imp


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    top_n: int,
    save_path: str
) -> None:
    if importance_df.empty:
        return

    plot_df = importance_df.head(top_n).sort_values("abs_value", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature"], plot_df["value"])
    plt.xlabel("Coefficient / Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance: {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_results_table(results_df: pd.DataFrame, save_path: str) -> None:
    results_df.to_csv(save_path, index=False)