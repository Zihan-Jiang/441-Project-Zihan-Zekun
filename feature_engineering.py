import pandas as pd
import numpy as np

# Load the cleaned stock dataset
df = pd.read_csv("stock_data.csv")

# Convert the Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Sort the data by ticker and date
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Calculate daily return for each stock
df["Return"] = df.groupby("Ticker")["Close"].pct_change()

# Calculate moving averages with different time windows
df["MA5"] = df.groupby("Ticker")["Close"].transform(
    lambda x: x.rolling(window=5).mean()
)

df["MA10"] = df.groupby("Ticker")["Close"].transform(
    lambda x: x.rolling(window=10).mean()
)

df["MA20"] = df.groupby("Ticker")["Close"].transform(
    lambda x: x.rolling(window=20).mean()
)

df["MA60"] = df.groupby("Ticker")["Close"].transform(
    lambda x: x.rolling(window=60).mean()
)

# Calculate rolling volatility based on daily returns
df["Volatility"] = df.groupby("Ticker")["Return"].transform(
    lambda x: x.rolling(window=20).std()
)

# Calculate percentage change in trading volume
df["Volume_change"] = df.groupby("Ticker")["Volume"].pct_change()

# Calculate intraday price spread
df["High_Low_Spread"] = (df["High"] - df["Low"]) / df["Close"]

# Calculate open-close percentage change
df["Open_Close_Change"] = (df["Close"] - df["Open"]) / df["Open"]

# Create the prediction target:
# 1 if the next day's return is positive, otherwise 0
df["Target"] = (
    df.groupby("Ticker")["Close"].pct_change().shift(-1) > 0
).astype(int)

# Sort the final dataset by date and ticker for easier reading
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# Save the feature dataset without dropping missing values
df.to_csv("stock_features_dataset.csv", index=False)

# Print basic information
print("Feature engineering completed successfully.")
print(df.head())
print("\nColumns in the final dataset:")
print(df.columns.tolist())
print("\nDataset shape:")
print(df.shape)

print("\nMissing values in each column:")
print(df.isna().sum())

print("\nUnique tickers:")
print(df["Ticker"].unique())