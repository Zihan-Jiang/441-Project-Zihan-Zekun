from __future__ import annotations
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # returns
    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_5"] = out["Close"].pct_change(5)

    # candlestick structure
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_return"] = (out["Close"] - out["Open"]) / out["Open"]

    # moving averages
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_5_ratio"] = out["Close"] / out["ma_5"]
    out["ma_10_ratio"] = out["Close"] / out["ma_10"]

    # rolling volatility
    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["vol_10"] = out["ret_1"].rolling(10).std()

    # volume features
    out["vol_chg_1"] = out["Volume"].pct_change(1)
    out["vol_mean_5"] = out["Volume"].rolling(5).mean()
    out["vol_ratio_5"] = out["Volume"] / out["vol_mean_5"]

    # 清掉中间过渡列
    out = out.drop(columns=["ma_5", "ma_10", "vol_mean_5"])

    # 删掉 rolling / pct_change 产生的 NA
    out = out.dropna().reset_index(drop=True)
    return out


def get_feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_3",
        "ret_5",
        "hl_range",
        "oc_return",
        "ma_5_ratio",
        "ma_10_ratio",
        "vol_5",
        "vol_10",
        "vol_chg_1",
        "vol_ratio_5",
    ]