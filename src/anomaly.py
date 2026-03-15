from __future__ import annotations
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def detect_anomalies(
    df: pd.DataFrame,
    threshold: float = 2.0
) -> pd.DataFrame:
    out = df.copy()

    # Use existing feature columns as anomaly signals.
    out["volatility_z"] = zscore(out["vol_10"])
    out["volume_z"] = zscore(out["vol_ratio_5"])

    out["anomaly_score"] = 0.5 * out["volatility_z"].abs() + 0.5 * out["volume_z"].abs()
    out["is_anomaly"] = (out["anomaly_score"] >= threshold).astype(int)

    return out