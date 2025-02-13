import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta  # Use pandas_ta for technical analysis
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# Function to download stock data for Indian stocks
def get_stock_data(stock_symbol, start_date='2020-01-01', end_date='2025-01-01'):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to calculate Value at Risk (VaR) - 1 Day 95% Confidence Interval
def calculate_var(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

# Function to forecast using ARIMA
def forecast_arima(data, steps=5):
    model = ARIMA(data['Close'], order=(5, 1, 0))  # ARIMA(5,1,0) for simplicity
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Machine Learning model for portfolio risk prediction
def risk_prediction_model(stock_data):
    # Calculate Returns
    stock_data['Returns'] = stock_data['Close'].pct_change()  # Calculate returns
    stock_data = stock_data.dropna()
    
    # Technical indicators using pandas_ta
    stock_data['SMA'] = ta.sma(stock_data['Close'], length=14)  # Simple Moving Average
    stock_data['EMA'] = ta.ema(stock_data['Close'], length=14)  # Exponential Moving Average
    
    # Prepare features and labels
    stock_data = stock_data.dropna()  # Drop rows with missing values after indicator calculations
    features = stock_data[['SMA', 'EMA', 'Returns']].iloc[14:]  # Use features starting after the SMA/EMA calculation
    labels = stock_data['Returns'].iloc[14:]
    
    # Split into training and testing sets
    train_size = int(len(features) * 0.8)
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest Regression Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate Model
    rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
    return model, rmse

# Streamlit Dashboard
def run_dashboard():
    st.title("Stock Portfolio Real-Time Risk Prediction and Forecasting")

    # Stock Symbol Input
    stock_symbol = st.text_input("Enter Stock Symbol (e.g. TCS.NS for TCS India):", "TCS.NS")
    
    # Download Stock Data
    stock_data = get_stock_data(stock_symbol)
    st.subheader(f"Showing Data for {stock_symbol}")
    st.write(stock_data.tail())

    # Calculate Risk (VaR)
    stock_returns = stock_data['Close'].pct_change().dropna()
    var_95 = calculate_var(stock_returns, confidence_level=0.95)
    st.subheader("Value at Risk (VaR) Calculation")
    st.write(f"1-Day 95% VaR: {var_95:.2%}")

    # ARIMA Forecast
    forecast = forecast_arima(stock_data, steps=5)
    st.subheader("ARIMA Stock Price Forecast (Next 5 Days)")
    st.write(forecast)

    # Risk Prediction Model using Random Forest
    model, rmse = risk_prediction_model(stock_data)
    st.subheader(f"Risk Prediction Model - RMSE: {rmse:.4f}")
    st.write("Model trained with technical indicators and stock returns.")

    # Show Portfolio Risk Prediction
    st.subheader("Portfolio Risk Prediction (Next Day)")
    next_day_data = stock_data.iloc[-1:]  # Last day's data
    next_day_features = next_day_data[['SMA', 'EMA', 'Returns']]  # Get the required features
    next_day_scaled = StandardScaler().fit_transform(next_day_features)  # Scale the features
    predicted_risk = model.predict(next_day_scaled)  # Predict the risk
    st.write(f"Predicted Risk for Next Day: {predicted_risk[0]:.4f}")

# Run the Streamlit App
if __name__ == "__main__":
    run_dashboard()
