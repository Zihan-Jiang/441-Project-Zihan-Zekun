from __future__ import annotations
import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = "Date"
):
    train = df[df[date_col] <= train_end].copy()
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test = df[df[date_col] > val_end].copy()

    return train, val, test