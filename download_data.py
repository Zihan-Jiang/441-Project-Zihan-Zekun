import yfinance as yf
import pandas as pd

# List of stock tickers to download
tickers = ["SPY", "AAPL", "MSFT", "GOOG", "AMZN"]

# Define the time range for the data
start_date = "2010-01-01"
end_date = "2025-01-01"

# Download historical stock data from Yahoo Finance
data = yf.download(
    tickers,
    start=start_date,
    end=end_date
)

# Convert the multi-index columns into a long (tidy) format
data = data.stack(level=1).reset_index()

# Rename the column that contains ticker symbols
data = data.rename(columns={
    "level_1": "Ticker"
})

# Save the cleaned dataset to a CSV file
data.to_csv("stock_data.csv", index=False)

# Print a preview of the dataset
print("Download finished")
print(data.head())