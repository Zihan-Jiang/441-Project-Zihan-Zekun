from __future__ import annotations
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score using only past information.
    The current day is compared with the mean and standard deviation
    from the previous rolling window.
    """
    rolling_mean = series.shift(1).rolling(window=window, min_periods=20).mean()
    rolling_std = series.shift(1).rolling(window=window, min_periods=20).std(ddof=0)

    return (series - rolling_mean) / rolling_std


def detect_anomalies(
    df: pd.DataFrame,
    threshold: float = 2.0,
    window: int = 60
) -> pd.DataFrame:
    """
    Detect anomalous market days using rolling z-scores.

    Signals:
    - return_z: unusual daily return
    - volatility_z: unusual short-term volatility
    - volume_z: unusual volume ratio

    A day is marked as anomalous if the average absolute z-score
    is greater than or equal to the selected threshold.
    """
    out = df.copy()

    out["return_z"] = rolling_zscore(out["ret_1"], window=window)
    out["volatility_z"] = rolling_zscore(out["vol_10"], window=window)
    out["volume_z"] = rolling_zscore(out["vol_ratio_5"], window=window)

    out["anomaly_score"] = (
        out["return_z"].abs()
        + out["volatility_z"].abs()
        + out["volume_z"].abs()
    ) / 3

    out["is_anomaly"] = (out["anomaly_score"] >= threshold).astype(int)

    return out