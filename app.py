import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data, get_single_ticker_data
from src.target import make_binary_target
from src.features import build_features, get_feature_columns
from src.split import time_based_split
from src.compare import run_model_comparison
from src.backtest import rolling_backtest
from src.anomaly import detect_anomalies
from src.visualize import extract_feature_importance
from src.config import MODEL_NAMES


st.set_page_config(
    page_title="Stock Prediction Dashboard",
    layout="wide"
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_raw_data(path: str) -> pd.DataFrame:
    return load_data(path)


def prepare_single_ticker_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df_ticker = get_single_ticker_data(df, ticker)
    df_ticker = make_binary_target(df_ticker, price_col="Close")
    df_ticker = build_features(df_ticker)
    return df_ticker


def run_pipeline(
    raw_df: pd.DataFrame,
    ticker: str,
    train_end: str,
    val_end: str,
    anomaly_threshold: float
):
    df_ticker = prepare_single_ticker_data(raw_df, ticker)

    feature_cols = get_feature_columns()
    target_col = "Target"

    train_df, val_df, test_df = time_based_split(
        df_ticker,
        train_end=train_end,
        val_end=val_end
    )

    results_df, fitted_models, detailed_results = run_model_comparison(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_names=MODEL_NAMES,
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    bt_results = rolling_backtest(
        df=df_ticker,
        feature_cols=feature_cols,
        target_col=target_col,
        model=best_model,
        train_window=756,
        test_window=63
    )

    anomaly_df = detect_anomalies(df_ticker, threshold=anomaly_threshold)

    return {
        "df_ticker": df_ticker,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "results_df": results_df,
        "fitted_models": fitted_models,
        "detailed_results": detailed_results,
        "best_model_name": best_model_name,
        "bt_results": bt_results,
        "anomaly_df": anomaly_df,
    }


def plot_roc_streamlit(detailed_results: dict, split_name: str = "test"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, result_dict in detailed_results.items():
        metrics = result_dict[split_name]
        if metrics["fpr"] is not None and metrics["tpr"] is not None:
            auc_value = metrics["roc_auc"]
            ax.plot(metrics["fpr"], metrics["tpr"], label=f"{model_name} (AUC={auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves ({split_name.title()})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def plot_pr_streamlit(detailed_results: dict, split_name: str = "test"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, result_dict in detailed_results.items():
        metrics = result_dict[split_name]
        if metrics["pr_recall_curve"] is not None and metrics["pr_precision_curve"] is not None:
            ap_value = metrics["avg_precision"]
            ax.plot(
                metrics["pr_recall_curve"],
                metrics["pr_precision_curve"],
                label=f"{model_name} (AP={ap_value:.3f})"
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves ({split_name.title()})")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def plot_confusion_matrix_streamlit(detailed_results: dict, model_name: str, split_name: str = "test"):
    from sklearn.metrics import ConfusionMatrixDisplay

    metrics = detailed_results[model_name][split_name]
    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name} ({split_name.title()})")
    st.pyplot(fig)


def plot_feature_importance_streamlit(model, feature_cols: list[str], model_name: str, top_n: int = 10):
    importance_df = extract_feature_importance(model, feature_cols)

    if importance_df.empty:
        st.info(f"{model_name} does not provide direct feature importance.")
        return

    plot_df = importance_df.head(top_n).sort_values("abs_value", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(plot_df["feature"], plot_df["value"])
    ax.set_xlabel("Coefficient / Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Features: {model_name}")
    st.pyplot(fig)

    st.dataframe(importance_df, use_container_width=True)


def plot_backtest_streamlit(bt_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bt_results["test_end"], bt_results["roc_auc"], marker="o")
    ax.set_xlabel("Test End Date")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Rolling Backtest ROC-AUC Over Time")
    ax.grid(True)
    st.pyplot(fig)

    st.dataframe(bt_results, use_container_width=True)


def plot_anomaly_price_chart(anomaly_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(anomaly_df["Date"], anomaly_df["Close"], label="Close")

    anomalies = anomaly_df[anomaly_df["is_anomaly"] == 1]
    ax.scatter(
        anomalies["Date"],
        anomalies["Close"],
        marker="o",
        s=20,
        label="Anomaly"
    )

    ax.set_title("Price Series with Anomaly Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Controls")

raw_df = load_raw_data("data/stock_data.csv")
ticker_list = sorted(raw_df["Ticker"].unique().tolist())

ticker = st.sidebar.selectbox("Select ticker", ticker_list, index=0)

train_end = st.sidebar.text_input("Train end date", "2020-12-31")
val_end = st.sidebar.text_input("Validation end date", "2022-12-31")

anomaly_threshold = st.sidebar.slider(
    "Anomaly threshold",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1
)

selected_model_for_display = st.sidebar.selectbox(
    "Model for CM / Feature Importance",
    MODEL_NAMES,
    index=0
)

run_button = st.sidebar.button("Run analysis")


# -----------------------------
# Main page
# -----------------------------
st.title("Short-Term Stock Direction Prediction Dashboard")
st.markdown(
    """
    This dashboard runs a single-ticker ML pipeline for next-day up/down prediction,
    compares classical models, shows backtesting performance, and highlights anomalous market days.
    """
)

if run_button:
    with st.spinner("Running pipeline..."):
        output = run_pipeline(
            raw_df=raw_df,
            ticker=ticker,
            train_end=train_end,
            val_end=val_end,
            anomaly_threshold=anomaly_threshold
        )

    df_ticker = output["df_ticker"]
    train_df = output["train_df"]
    val_df = output["val_df"]
    test_df = output["test_df"]
    feature_cols = output["feature_cols"]
    results_df = output["results_df"]
    fitted_models = output["fitted_models"]
    detailed_results = output["detailed_results"]
    best_model_name = output["best_model_name"]
    bt_results = output["bt_results"]
    anomaly_df = output["anomaly_df"]

    if selected_model_for_display not in fitted_models:
        selected_model_for_display = best_model_name

    st.success(f"Finished. Best model by test ROC-AUC: {best_model_name}")

    # -----------------------------
    # Section 1: Data overview
    # -----------------------------
    st.header("1. Data Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", ticker)
    c2.metric("Rows after features", len(df_ticker))
    c3.metric("Train rows", len(train_df))
    c4.metric("Test rows", len(test_df))

    st.subheader("Processed Data Preview")
    st.dataframe(df_ticker.head(20), use_container_width=True)

    st.subheader("Feature Columns")
    st.write(feature_cols)

    # -----------------------------
    # Section 2: Model comparison
    # -----------------------------
    st.header("2. Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    # -----------------------------
    # Section 3: ROC / PR curves
    # -----------------------------
    st.header("3. Performance Curves")

    col1, col2 = st.columns(2)
    with col1:
        plot_roc_streamlit(detailed_results, split_name="test")
    with col2:
        plot_pr_streamlit(detailed_results, split_name="test")

    # -----------------------------
    # Section 4: Confusion matrix
    # -----------------------------
    st.header("4. Confusion Matrix")
    plot_confusion_matrix_streamlit(
        detailed_results=detailed_results,
        model_name=selected_model_for_display,
        split_name="test"
    )

    # -----------------------------
    # Section 5: Feature importance
    # -----------------------------
    st.header("5. Feature Importance / Coefficients")
    plot_feature_importance_streamlit(
        model=fitted_models[selected_model_for_display],
        feature_cols=feature_cols,
        model_name=selected_model_for_display,
        top_n=10
    )

    # -----------------------------
    # Section 6: Rolling backtest
    # -----------------------------
    st.header("6. Rolling Backtest")
    plot_backtest_streamlit(bt_results)

    # -----------------------------
    # Section 7: Anomaly detection
    # -----------------------------
    st.header("7. Anomaly Detection")

    c1, c2 = st.columns([2, 1])

    with c1:
        plot_anomaly_price_chart(anomaly_df)

    with c2:
        st.subheader("Top anomaly days")
        anomalies_only = anomaly_df[anomaly_df["is_anomaly"] == 1].copy()
        anomalies_only = anomalies_only.sort_values("anomaly_score", ascending=False)
        st.dataframe(
            anomalies_only[["Date", "Close", "vol_10", "vol_ratio_5", "anomaly_score"]].head(20),
            use_container_width=True
        )

else:
    st.info("Set parameters in the sidebar and click 'Run analysis'.")