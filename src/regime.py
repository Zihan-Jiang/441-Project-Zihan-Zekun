from __future__ import annotations
import pandas as pd


def assign_market_regime(date) -> str:
    """
    Assign a market regime based on date.
    This is a simple rule-based regime definition for interpretation.
    """
    date = pd.to_datetime(date)

    if date < pd.Timestamp("2020-01-01"):
        return "Pre-COVID"
    elif date < pd.Timestamp("2022-01-01"):
        return "COVID / High Volatility"
    elif date < pd.Timestamp("2023-01-01"):
        return "2022 Bear Market / Rate Hikes"
    else:
        return "Recovery / AI-driven Market"


def summarize_backtest_by_regime(bt_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize rolling backtest performance by market regime.
    """
    out = bt_results.copy()
    out["regime"] = out["test_end"].apply(assign_market_regime)

    summary = (
        out.groupby("regime")
        .agg(
            n_windows=("roc_auc", "count"),
            avg_accuracy=("accuracy", "mean"),
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
            avg_f1=("f1", "mean"),
            avg_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )

    regime_order = [
        "Pre-COVID",
        "COVID / High Volatility",
        "2022 Bear Market / Rate Hikes",
        "Recovery / AI-driven Market",
    ]

    summary["regime"] = pd.Categorical(
        summary["regime"],
        categories=regime_order,
        ordered=True
    )

    summary = summary.sort_values("regime").reset_index(drop=True)
    return summary