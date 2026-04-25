# Interpretable Stock Direction Prediction and Anomaly Detection with Machine Learning

## 1. Project Description

This project builds an interpretable machine learning web application for short-term stock direction prediction and anomaly detection.

The prediction task is formulated as a supervised binary classification problem. Given historical daily OHLCV stock data and technical indicators, the model predicts whether the adjusted closing price of a selected stock will go up or down on the next trading day.

The goal of this project is not to guarantee profitable trading. Instead, the project aims to evaluate how classical machine learning models perform on noisy and non-stationary financial time series. The dashboard also provides model comparison, rolling backtesting, feature importance, market regime analysis, anomaly detection, and statistical analysis of daily returns.

## 2. Data Source

The stock price data comes from Yahoo Finance and is downloaded using the `yfinance` Python package.

The project uses historical daily OHLCV data for the following assets:

- SPY
- AAPL
- MSFT
- GOOG
- AMZN

The raw dataset includes daily open, high, low, close, adjusted close, and trading volume. Technical indicators are generated from the raw price and volume data.

## 3. Main Features

- Short-term stock direction prediction
- Time-aware train / validation / test split
- Classical machine learning model comparison
- Rolling backtesting
- Market regime analysis
- Feature importance for interpretability
- Anomaly detection using rolling z-scores
- Statistical analysis of daily returns
- Interactive Streamlit dashboard

## 4. Models Used

This project uses classical machine learning models only. No deep learning models are used.

The models include:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- Gradient Boosting

## 5. Evaluation Metrics

The project reports the following evaluation metrics:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Majority-class baseline accuracy

Because stock up/down labels can be imbalanced, accuracy alone may be misleading. Therefore, the project also reports balanced accuracy, ROC-AUC, prediction positive rate, and majority-class baseline accuracy.

The best model is selected using validation ROC-AUC. The test set is used only for final evaluation.

## 6. Time-Series Validation

To avoid look-ahead bias, the project uses chronological train / validation / test splitting.

The dashboard also includes rolling backtesting to evaluate how model performance changes over time. This is important because financial markets are noisy and non-stationary.

## 7. Anomaly Detection

The anomaly detection module uses rolling z-scores based only on past information.

The anomaly score is based on:

- Daily return
- Short-term volatility
- Volume ratio

A trading day is flagged as anomalous if its average absolute rolling z-score exceeds the selected threshold.

## 8. Market Regime Analysis

The dashboard summarizes rolling backtest performance across different market regimes:

- Pre-COVID
- COVID / High Volatility
- 2022 Bear Market / Rate Hikes
- Recovery / AI-driven Market

This helps evaluate whether model performance changes under different market conditions.

## 9. Required Packages

The required Python packages are listed in `requirements.txt`.

Main packages include:

- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- plotly
- streamlit
- yfinance

To install all required packages, run:

```bash
pip install -r requirements.txt
```

## 10. How to Run the Code

First, install the required packages:

```bash
pip install -r requirements.txt
```

Then run the Streamlit web application:

```bash
streamlit run app.py
```

If the command above does not work, use:

```bash
python -m streamlit run app.py
```

After running the command, open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 11. Project Structure

```text
.
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── ReadMe.txt
├── data/
│   ├── readme_data.txt
│   └── stock_data.csv
├── outputs/
├── scripts/
│   └── download_data.py
└── src/
    ├── anomaly.py
    ├── backtest.py
    ├── compare.py
    ├── config.py
    ├── data_loader.py
    ├── evaluate.py
    ├── features.py
    ├── models.py
    ├── regime.py
    ├── split.py
    ├── statistics.py
    ├── target.py
    └── visualize.py
```

## 12. Limitations

Short-term stock direction prediction is difficult using only daily OHLCV data and technical indicators. Financial markets are noisy, non-stationary, and affected by many external factors that are not fully captured in this dataset.

The ROC-AUC values are often close to 0.5, which suggests that classical machine learning models have limited predictive power for next-day stock direction prediction. However, the project still provides value by comparing models, evaluating performance over time, analyzing market regimes, identifying important features, and detecting anomalous market days.

## 13. Web Application

The final product is an interactive Streamlit dashboard. Users can select a ticker, choose train and validation dates, run model comparison, inspect feature importance, evaluate rolling backtest performance, analyze market regimes, detect anomalous market days, and view statistical analysis results.

## 14. Future Work

Possible future improvements include:

- Adding macroeconomic variables
- Adding market index and sector-level features
- Testing longer prediction horizons
- Improving feature engineering
- Adding permutation importance for model interpretation
- Deploying the Streamlit dashboard online