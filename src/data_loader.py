from __future__ import annotations
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates()
    return df


def get_single_ticker_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ticker_df = df[df["Ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
    return ticker_df


def basic_data_check(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Date range:", df["Date"].min(), "to", df["Date"].max())
    print("Missing values:\n", df.isna().sum())