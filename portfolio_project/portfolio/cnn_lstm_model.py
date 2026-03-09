"""
CNN + LSTM Hybrid Model for Stock Price Prediction
===================================================
Deep Learning model combining Convolutional Neural Networks
and Long Short-Term Memory networks for stock price forecasting.

Supported Markets:
- BTC/USD (using BTC-USD ticker)
- Indian Stocks (using NSE format, e.g., RELIANCE.NS)

Functions:
- fetch_stock_data(ticker, period)
- clean_data(df)
- create_sequences(data, look_back)
- build_cnn_lstm_model(input_shape)
- train_model(model, X_train, y_train)
- predict_future_prices(model, scaler, last_sequence, n_predictions)
- evaluate_model(model, X_test, y_test, scaler)
- run_cnn_lstm_forecast(ticker, period)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Try to import TensorFlow, if not available, use alternative
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
    print("TensorFlow loaded successfully:", tf.__version__)

except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow import error:", str(e))


# Cache directory for storing fetched data
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')
CACHE_DURATION = 3600  # 1 hour in seconds


def _get_cache_path(ticker):
    """Generate cache file path for a ticker."""
    return os.path.join(CACHE_DIR, f"{ticker.replace('.', '_').replace('-', '_')}.json")


def _get_cached_data(ticker):
    """Get cached data if available and not expired."""
    cache_path = _get_cache_path(ticker)
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        import time
        if time.time() - cache_data['timestamp'] > CACHE_DURATION:
            return None
        
        return cache_data['data']
    except:
        return None


def _save_cached_data(ticker, data):
    """Save data to cache."""
    cache_path = _get_cache_path(ticker)
    try:
        import time
        cache_data = {
            'timestamp': time.time(),
            'data': data
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
    except:
        pass  # Silently fail if caching fails


def fetch_stock_data(ticker, period="4y"):
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker: Stock/crypto ticker symbol (e.g., 'BTC-USD', 'RELIANCE.NS')
        period: Data period to fetch (default: 4 years)
    
    Returns:
        DataFrame with OHLCV data or None on error
    """
    # Try to get cached data first
    cached = _get_cached_data(ticker)
    if cached is not None:
        df = pd.DataFrame(cached)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period)
        
        if df is None or df.empty:
            return None
        
        # Save to cache
        cache_data = df.reset_index().to_dict(orient='records')
        for record in cache_data:
            if 'Date' in record:
                record['Date'] = str(record['Date'])
        _save_cached_data(ticker, cache_data)
        
        return df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def clean_data(df):
    """
    Clean the input DataFrame for CNN+LSTM modeling.
    
    Steps:
    - Remove missing values
    - Handle NaN values
    - Convert timestamps properly
    - Ensure required columns exist
    - Sort data by date
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Cleaned DataFrame or None on error
    """
    try:
        if df is None or df.empty:
            return None
        
        # Required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return None
        
        # Create a copy
        df_clean = df.copy()
        
        # Convert index to datetime if needed
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Remove duplicate dates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Get only required columns
        df_clean = df_clean[required_cols]
        
        # Remove missing values
        df_clean = df_clean.dropna()
        
        # Fill any remaining NaN with forward fill then backward fill
        if df_clean.isna().any().any():
            df_clean = df_clean.ffill().bfill()
        
        # Remove any infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        # Ensure we have enough data points
        if len(df_clean) < 100:  # Need at least 100 days of data
            return None
        
        return df_clean
    
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None


def create_sequences(data, look_back=60):
    """
    Create sequences for CNN+LSTM training.
    Uses last 'look_back' days to predict next day.
    
    Args:
        data: numpy array of normalized data
        look_back: Number of timesteps to use for prediction
    
    Returns:
        X: sequences (samples, timesteps, features)
        y: target values
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 3])  # Close price is at index 3
    
    return np.array(X), np.array(y)


def build_cnn_lstm_model(input_shape):
    """
    Build CNN + LSTM hybrid deep learning model.
    
    Architecture:
    - Conv1D layer with 64 filters
    - MaxPooling1D
    - Dropout
    - LSTM layer with 100 units
    - Dropout
    - Dense layer with 50 units
    - Output layer with 1 unit
    
    Args:
        input_shape: Tuple (timesteps, features)
    
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        return None
    
    model = Sequential([
        # Conv1D layer
        Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=input_shape
        ),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layer
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(units=50, activation='relu'),
        Dense(units=1)
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    return model


def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    """
    Train the CNN+LSTM model.
    
    Args:
        model: Compiled Keras model
        X_train: Training sequences
        y_train: Training targets
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Training history or None on error
    """
    if not TF_AVAILABLE or model is None:
        return None
    
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        return history
    except Exception as e:
        print(f"Error training model: {e}")
        return None


def predict_future_prices(model, scaler, last_sequence, n_predictions=5):
    """
    Predict future prices using the trained model.
    
    Args:
        model: Trained Keras model
        scaler: Fitted MinMaxScaler
        last_sequence: Last sequence of normalized data
        n_predictions: Number of days to predict
    
    Returns:
        List of predicted prices
    """
    if not TF_AVAILABLE or model is None:
        return []
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_predictions):
        # Reshape for prediction: (1, timesteps, features)
        current_sequence_reshaped = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next day
        pred = model.predict(current_sequence_reshaped, verbose=0)
        
        # Append prediction
        predictions.append(float(pred[0, 0]))
        
        # Update sequence for next prediction
        # Create a new row with the prediction for Close price (index 3)
        new_row = current_sequence[-1].copy()
        new_row[3] = pred[0, 0]  # Update Close price
        
        # Shift all values and add new prediction
        # For other columns, we'll use a simple approach
        for i in range(len(current_sequence) - 1):
            current_sequence[i] = current_sequence[i + 1]
        current_sequence[-1] = new_row
    
    # Inverse transform predictions to get actual prices
    # Create a dummy array with the same shape as original data
    dummy = np.zeros((len(predictions), 5))  # 5 features
    dummy[:, 3] = predictions  # Close price is at index 3
    
    # Inverse transform
    predicted_prices = scaler.inverse_transform(dummy)[:, 3]
    
    return predicted_prices.tolist()


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Test targets
        scaler: Fitted MinMaxScaler
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not TF_AVAILABLE or model is None:
        return {
            'mae': None,
            'rmse': None,
            'r2': None,
            'accuracy': None
        }
    
    try:
        # Predict on test data
        y_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform to get actual prices
        y_test_actual = scaler.inverse_transform(
            np.hstack([np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)])
        )[:, 4]
        
        y_pred_actual = scaler.inverse_transform(
            np.hstack([np.zeros((len(y_pred), 4)), y_pred.reshape(-1, 1)])
        )[:, 4]
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        # Calculate accuracy (as per requirements)
        mean_price = np.mean(y_test_actual)
        if mean_price > 0:
            accuracy = (1 - rmse / mean_price) * 100
            accuracy = max(0, min(100, accuracy))  # Clamp between 0 and 100
        else:
            accuracy = 0
        
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 2),
            'accuracy': round(accuracy, 2)
        }
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {
            'mae': None,
            'rmse': None,
            'r2': None,
            'accuracy': None
        }


def run_cnn_lstm_forecast(ticker, period="4y", forecast_days=5):
    """
    Main function to run CNN+LSTM forecast for a given ticker.
    
    Args:
        ticker: Stock/crypto ticker symbol
        period: Historical data period to fetch
        forecast_days: Number of days to forecast
    
    Returns:
        Dictionary with predictions and metrics or error information
    """
    if not TF_AVAILABLE:
        return {
            'success': False,
            'error': 'TensorFlow is not available. Please install tensorflow.'
        }
    
    try:
        # Step 1: Fetch stock data
        df = fetch_stock_data(ticker, period)
        
        if df is None or df.empty:
            return {
                'success': False,
                'error': f'Could not fetch data for {ticker}. Please check the symbol.'
            }
        
        # Step 2: Clean data
        df_clean = clean_data(df)
        
        if df_clean is None or len(df_clean) < 100:
            return {
                'success': False,
                'error': 'Insufficient data for CNN+LSTM modeling. Need at least 100 days of data.'
            }
        
        # Step 3: Prepare data for training
        # Use all features: Open, High, Low, Close, Volume
        data = df_clean[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        look_back = 60  # Use last 60 days to predict next day
        X, y = create_sequences(scaled_data, look_back)
        
        if len(X) < 50:
            return {
                'success': False,
                'error': 'Insufficient data for sequence creation.'
            }
        
        # Split data: 80% train, 20% test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Step 4: Build model
        model = build_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        if model is None:
            return {
                'success': False,
                'error': 'Failed to build CNN+LSTM model.'
            }
        
        # Step 5: Train model
        history = train_model(model, X_train, y_train, epochs=20, batch_size=32)
        
        if history is None:
            return {
                'success': False,
                'error': 'Failed to train CNN+LSTM model.'
            }
        
        # Step 6: Evaluate model
        metrics = evaluate_model(model, X_test, y_test, scaler)
        
        # Step 7: Predict future prices
        # Get the last sequence for prediction
        last_sequence = scaled_data[-look_back:]
        predictions = predict_future_prices(model, scaler, last_sequence, forecast_days)
        
        if len(predictions) == 0:
            return {
                'success': False,
                'error': 'Failed to generate predictions.'
            }
        
        # Generate future dates
        last_date = df_clean.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        date_strings = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        # Prepare historical data for output (last 180 days)
        historical_dates = df_clean.index.strftime('%Y-%m-%d').tolist()[-180:]
        historical_prices = df_clean['Close'].values.tolist()[-180:]
        
        return {
            'success': True,
            'historical_dates': historical_dates,
            'historical_prices': historical_prices,
            'predictions': predictions,
            'prediction_dates': date_strings,
            'accuracy': metrics['accuracy'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        }
    
    except Exception as e:
        print(f"Error in run_cnn_lstm_forecast: {e}")
        return {
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }


def convert_symbol(symbol):
    """
    Convert Indian stock symbol to NSE format.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS', 'HDFCBANK')
    
    Returns:
        Ticker in appropriate format
    """
    # Remove .NS or .BO if already present
    symbol = symbol.upper().strip().replace('.NS', '').replace('.BO', '')
    
    # If it contains a dash (like BTC-USD), it's likely crypto
    if '-' in symbol and symbol != 'BTC-USD':
        # Could be a forex pair or something else
        pass
    
    # Common Indian stock mappings - check if it's an Indian stock
    indian_stocks = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'HDFC', 'ICICIBANK', 'SBIN',
        'AXISBANK', 'KOTAKBANK', 'INDUSINDBK', 'BANKBARODA', 'PNB',
        'BAJFINANCE', 'MARUTI', 'TATAMOTORS', 'WIPRO', 'TECHM', 'HCLTECH',
        'ADANIPORTS', 'ASIANPAINT', 'BRITANNIA', 'CIPLA', 'DRREDDY',
        'EICHERMOT', 'GRASIM', 'HINDUNILVR', 'ITC', 'JSWSTEEL', 'LT',
        'M&M', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'SUNPHARMA',
        'TATASTEEL', 'ULTRACEMCO', 'UPL'
    ]
    
    # Check if symbol is in Indian stocks list or ends with common NSE patterns
    if symbol in indian_stocks or not symbol.endswith('-'):
        # Assume it's an Indian stock if not crypto
        if symbol != 'BTC-USD':
            return symbol + '.NS'
    
    return symbol

