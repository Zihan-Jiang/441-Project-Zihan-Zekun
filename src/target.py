from __future__ import annotations
import pandas as pd


def make_binary_target(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Target = 1 if next day's Close > today's Close else 0
    """
    out = df.copy()
    out["future_close"] = out[price_col].shift(-1)
    out["Target"] = (out["future_close"] > out[price_col]).astype(int)

    # 最后一行没有 future_close，删掉
    out = out.dropna(subset=["future_close"]).reset_index(drop=True)
    out = out.drop(columns=["future_close"])
    return out