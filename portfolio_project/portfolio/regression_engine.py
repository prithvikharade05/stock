"""
Regression Engine for Stock Price Prediction
==============================================
Provides functions for predicting stock prices using Linear and Logistic Regression.

Features:
- Lag 1 return
- Lag 5 return
- Moving Average (20)
- Moving Average (50)
- Volatility (20-day std)
- RSI (14-day)
"""

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
import math
warnings.filterwarnings('ignore')


def fetch_stock_data(symbol, period="3y"):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'RELIANCE')
        period: Data period to fetch (default: 3 years)
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        ticker = yf.Ticker(symbol + ".NS")
        hist = ticker.history(period=period)
        
        if hist is None or hist.empty or 'Close' not in hist.columns:
            return None
        
        # Ensure we have enough data
        if len(hist) < 100:
            return None
        
        return hist
    except Exception as e:
        return None


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_features(df):
    """
    Generate features for prediction.
    
    Args:
        df: DataFrame with Close prices
    
    Returns:
        DataFrame with features and target variables
    """
    data = df.copy()
    
    # Ensure we have Close prices
    if 'Close' not in data.columns:
        return None
    
    close = data['Close']
    
    # Feature: Lag 1 return (1-day return)
    data['return_1d'] = close.pct_change(1)
    
    # Feature: Lag 5 return (5-day return)
    data['return_5d'] = close.pct_change(5)
    
    # Feature: Moving Average 20
    data['ma20'] = close.rolling(window=20).mean()
    
    # Feature: Moving Average 50
    data['ma50'] = close.rolling(window=50).mean()
    
    # Feature: Volatility (20-day standard deviation)
    data['volatility_20'] = close.rolling(window=20).std()
    
    # Feature: RSI
    data['rsi'] = calculate_rsi(close, 14)
    
    # Target: Next day return (for training)
    data['target_return'] = close.pct_change(1).shift(-1)
    
    # Target: Next day direction (1 = up, 0 = down) - for logistic
    data['target_direction'] = (data['target_return'] > 0).astype(int)
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data


def prepare_training_data(data):
    """
    Prepare feature matrix and target vector for training.
    
    Args:
        data: DataFrame from create_features()
    
    Returns:
        Tuple: (X, y_linear, y_logistic, feature_names)
    """
    feature_columns = ['return_1d', 'return_5d', 'ma20', 'ma50', 'volatility_20', 'rsi']
    
    # Check if we have enough data
    if len(data) < 50:
        return None, None, None, None
    
    X = data[feature_columns].values
    
    # For linear regression: predict actual return
    y_linear = data['target_return'].values
    
    # For logistic regression: predict direction
    y_logistic = data['target_direction'].values
    
    return X, y_linear, y_logistic, feature_columns


def train_linear_model(X, y):
    """
    Train Linear Regression model.
    
    Args:
        X: Feature matrix
        y: Target returns
    
    Returns:
        Tuple: (model, scaler) or (None, None) if failed
    """
    try:
        # Use time-series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        return model, scaler
    except Exception as e:
        return None, None


def train_logistic_model(X, y):
    """
    Train Logistic Regression model.
    
    Args:
        X: Feature matrix
        y: Target directions
    
    Returns:
        Tuple: (model, scaler) or (None, None) if failed
    """
    try:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        return model, scaler
    except Exception as e:
        return None, None


def get_latest_features(data):
    """
    Extract latest features for prediction.
    
    Args:
        data: DataFrame with features
    
    Returns:
        Feature vector for prediction
    """
    feature_columns = ['return_1d', 'return_5d', 'ma20', 'ma50', 'volatility_20', 'rsi']
    
    # Get the last row
    latest = data[feature_columns].iloc[-1].values
    
    return latest.reshape(1, -1)


def predict_next_two_days(model, scaler, latest_features, model_type="linear"):
    """
    Predict next 2 days prices/directions.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        latest_features: Latest feature vector
        model_type: "linear" or "logistic"
    
    Returns:
        List of predictions
    """
    predictions = []
    
    try:
        # Scale features
        X_scaled = scaler.transform(latest_features)
        
        if model_type == "linear":
            # Predict returns
            pred_return = model.predict(X_scaled)[0]
            predictions.append(pred_return)
            # For day 2, use same prediction (simple approach)
            predictions.append(pred_return)
        else:
            # Predict probabilities
            proba = model.predict_proba(X_scaled)[0]
            # Probability of going up
            prob_up = proba[1] if len(proba) > 1 else 0.5
            predictions.append(prob_up)
            # For day 2, use same probability
            predictions.append(prob_up)
            
    except Exception as e:
        return [None, None]
    
    return predictions


def predict_stock(symbol, model_type="linear"):
    """
    Main function to predict stock prices.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'RELIANCE')
        model_type: "linear" or "logistic"
    
    Returns:
        Dictionary with prediction results
    """
    result = {
        "symbol": symbol,
        "model_type": model_type,
        "success": False,
        "prediction_day_1": None,
        "prediction_day_2": None,
        "probability_day_1": None,
        "probability_day_2": None,
        "historical_prices": [],
        "predicted_points": [],
        "error": None
    }
    
    try:
        # Step 1: Fetch data
        df = fetch_stock_data(symbol)
        if df is None:
            result["error"] = "Failed to fetch stock data. Please check the symbol."
            return result
        
        # Step 2: Create features
        data = create_features(df)
        if data is None or len(data) < 50:
            result["error"] = "Insufficient data for prediction. Need at least 50 trading days."
            return result
        
        # Step 3: Prepare training data
        X, y_linear, y_logistic, feature_names = prepare_training_data(data)
        if X is None:
            result["error"] = "Failed to prepare training data."
            return result
        
        # Step 4: Train model
        if model_type == "linear":
            model, scaler = train_linear_model(X, y_linear)
        else:
            model, scaler = train_logistic_model(X, y_logistic)
        
        if model is None or scaler is None:
            result["error"] = "Failed to train model."
            return result
        
        # Step 5: Get latest features and predict
        latest_features = get_latest_features(data)
        predictions = predict_next_two_days(model, scaler, latest_features, model_type)
        
        # Step 6: Calculate predicted prices
        current_price = float(df['Close'].iloc[-1])
        
        if model_type == "linear":
            # Convert returns to prices
            pred_price_1 = current_price * (1 + predictions[0])
            pred_price_2 = current_price * (1 + predictions[1])
            
            result["prediction_day_1"] = round(pred_price_1, 2)
            result["prediction_day_2"] = round(pred_price_2, 2)
        else:
            # Probabilities
            result["probability_day_1"] = round(predictions[0] * 100, 2)
            result["probability_day_2"] = round(predictions[1] * 100, 2)
        
        # Step 7: Prepare historical prices for chart
        # Get last 60 days of data
        recent_data = df['Close'].tail(60)
        result["historical_prices"] = [
            {"date": str(date.date()), "price": float(price)}
            for date, price in recent_data.items()
        ]
        
        # Step 8: Prepare predicted points
        last_date = df.index[-1]
        # Next business days
        next_dates = []
        current_idx = len(recent_data) - 1
        
        for i in range(1, 3):
            # Simple approach: add days
            next_date = last_date + pd.Timedelta(days=i)
            # Skip weekends
            while next_date.dayofweek >= 5:
                next_date = next_date + pd.Timedelta(days=1)
            next_dates.append(str(next_date.date()))
        
        if model_type == "linear":
            result["predicted_points"] = [
                {"date": next_dates[0], "price": result["prediction_day_1"]},
                {"date": next_dates[1], "price": result["prediction_day_2"]}
            ]
        else:
            result["predicted_points"] = [
                {"date": next_dates[0], "probability": result["probability_day_1"]},
                {"date": next_dates[1], "probability": result["probability_day_2"]}
            ]
        
        result["current_price"] = round(current_price, 2)
        result["success"] = True
        
    except Exception as e:
        result["error"] = f"Error: {str(e)}"
    
    return result

