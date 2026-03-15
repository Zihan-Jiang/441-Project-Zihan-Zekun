from src.data_loader import load_data, get_single_ticker_data, basic_data_check
from src.target import make_binary_target
from src.features import build_features, get_feature_columns
from src.split import time_based_split
from src.backtest import rolling_backtest
from src.anomaly import detect_anomalies
from src.compare import run_model_comparison
from src.visualize import (
    ensure_dir,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix_for_model,
    extract_feature_importance,
    plot_feature_importance,
    save_results_table,
)
from src.config import MODEL_NAMES


def main():
    # -----------------------------
    # 0) output folders
    # -----------------------------
    ensure_dir("outputs/figures")
    ensure_dir("outputs/tables")

    # -----------------------------
    # 1) load data
    # -----------------------------
    df = load_data("data/stock_data.csv")
    basic_data_check(df)

    # -----------------------------
    # 2) choose one ticker
    # -----------------------------
    ticker = "AAPL"
    df_ticker = get_single_ticker_data(df, ticker)

    # -----------------------------
    # 3) target + features
    # -----------------------------
    df_ticker = make_binary_target(df_ticker, price_col="Close")
    df_ticker = build_features(df_ticker)

    feature_cols = get_feature_columns()
    target_col = "Target"

    # -----------------------------
    # 4) split
    # -----------------------------
    train_df, val_df, test_df = time_based_split(
        df_ticker,
        train_end="2020-12-31",
        val_end="2022-12-31"
    )

    print("\nTrain size:", train_df.shape)
    print("Val size:", val_df.shape)
    print("Test size:", test_df.shape)

    # -----------------------------
    # 5) compare models
    # -----------------------------
    results_df, fitted_models, detailed_results = run_model_comparison(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_names=MODEL_NAMES,
    )

    print("\nModel comparison results:")
    print(results_df)

    save_results_table(results_df, f"outputs/tables/{ticker}_model_comparison.csv")

    # -----------------------------
    # 6) plots: ROC / PR
    # -----------------------------
    plot_roc_curves(
        detailed_results=detailed_results,
        split_name="test",
        save_path=f"outputs/figures/{ticker}_roc_test.png"
    )

    plot_pr_curves(
        detailed_results=detailed_results,
        split_name="test",
        save_path=f"outputs/figures/{ticker}_pr_test.png"
    )

    # -----------------------------
    # 7) confusion matrix for best model
    # -----------------------------
    best_model_name = results_df.iloc[0]["model"]

    plot_confusion_matrix_for_model(
        detailed_results=detailed_results,
        model_name=best_model_name,
        split_name="test",
        save_path=f"outputs/figures/{ticker}_{best_model_name}_cm_test.png"
    )

    # -----------------------------
    # 8) feature importance
    # -----------------------------
    for model_name, model in fitted_models.items():
        importance_df = extract_feature_importance(model, feature_cols)

        if not importance_df.empty:
            importance_df.to_csv(
                f"outputs/tables/{ticker}_{model_name}_feature_importance.csv",
                index=False
            )

            plot_feature_importance(
                importance_df=importance_df,
                model_name=model_name,
                top_n=10,
                save_path=f"outputs/figures/{ticker}_{model_name}_feature_importance.png"
            )

    # -----------------------------
    # 9) rolling backtest for best model
    # -----------------------------
    best_model = fitted_models[best_model_name]

    bt_results = rolling_backtest(
        df=df_ticker,
        feature_cols=feature_cols,
        target_col=target_col,
        model=best_model,
        train_window=756,
        test_window=63
    )

    print("\nBacktest summary:")
    print(bt_results.head())
    print(bt_results.describe())

    bt_results.to_csv(f"outputs/tables/{ticker}_{best_model_name}_backtest.csv", index=False)

    # -----------------------------
    # 10) anomaly detection
    # -----------------------------
    anomaly_df = detect_anomalies(df_ticker, threshold=2.0)

    print("\nAnomaly sample:")
    print(
        anomaly_df.loc[
            anomaly_df["is_anomaly"] == 1,
            ["Date", "Close", "vol_10", "vol_ratio_5", "anomaly_score"]
        ].head(10)
    )

    anomaly_df.to_csv(f"outputs/tables/{ticker}_anomaly_output.csv", index=False)

    print("\nAll outputs saved under outputs/")

if __name__ == "__main__":
    main()