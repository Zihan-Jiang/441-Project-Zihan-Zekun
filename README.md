# 441-Project-Zihan-Zekun
## Summary
This project formulates short-term stock prediction as a supervised binary classification problem: predicting whether a stock’s adjusted closing price will go up or down on the next trading day based on historical OHLCV data and technical indicators. The project goals are (1) to build a reproducible machine learning pipeline for feature engineering, time-aware train/validation/test splitting, and rolling backtesting, (2) to compare classical models covered in the course—including Logistic Regression, SVM (linear and RBF kernel), and tree-based ensemble methods (e.g., Random Forest / Gradient Boosting)—using metrics such as accuracy, precision/recall, F1-score, ROC-AUC, and calibration, and (3) to develop an interpretable anomaly detection module based on volatility, volume, and/or model residuals with an adjustable threshold. The final product will be a Streamlit web application that allows users to select a ticker, train/evaluate models via time-series backtesting, visualize performance (ROC/PR curves and confusion matrix), inspect feature importance, and identify anomalous market days interactively.
## Data Source
yfinance
## Reference
