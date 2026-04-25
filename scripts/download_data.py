import yfinance as yf
import pandas as pd

def download_data():

    tickers = ["SPY", "AAPL", "MSFT", "GOOG", "AMZN"]

    start_date = "2010-01-01"
    end_date = "2025-01-01"

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date
    )

    data = data.stack(level=1).reset_index()

    data = data.rename(columns={
        "level_1": "Ticker"
    })

    data.to_csv("data/stock_data.csv", index=False)

    print("Download finished")
    print(data.head())


if __name__ == "__main__":
    download_data()