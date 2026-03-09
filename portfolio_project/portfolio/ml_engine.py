"""
ML Engine for Portfolio Clustering
===================================
Provides functions for extracting stock features, clustering stocks using KMeans,
and generating PCA-reduced coordinates for visualization.

Features:
- P/E Ratio
- Discount from 1Y High
- 1 Month Return
- 3 Month Return
- 6 Month Return
- LTP / 1Y High Ratio
"""

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def extract_features(stock_symbols):
    """
    Extract financial features for given stock symbols.
    
    Args:
        stock_symbols: List of stock ticker symbols (e.g., ['HDFCBANK', 'ICICIBANK'])
    
    Returns:
        Dictionary mapping stock_symbol -> feature_dict
        Feature dict contains: pe_ratio, discount_1y_high, return_1m, return_3m, 
                              return_6m, ltp_to_1y_high_ratio
    """
    features = {}
    
    for symbol in stock_symbols:
        try:
            ticker = yf.Ticker(symbol + ".NS")
            info = ticker.info
            
            # Get 1-year historical data
            hist = ticker.history(period="1y")
            
            if hist.empty or 'Close' not in hist.columns:
                continue
            
            close_prices = hist['Close']
            current_price = float(close_prices.iloc[-1])
            year_high = float(close_prices.max())
            year_low = float(close_prices.min())
            
            # Skip if no valid data
            if year_high is None or current_price is None or year_high == 0:
                continue
            
            # 1. P/E Ratio
            pe_ratio = info.get('trailingPE')
            if pe_ratio is None or pe_ratio <= 0 or not np.isfinite(pe_ratio):
                pe_ratio = None
            
            # 2. Discount from 1Y High (percentage)
            discount_1y_high = ((year_high - current_price) / year_high) * 100
            
            # 3. Returns calculation
            # 1 Month Return
            if len(close_prices) >= 21:  # ~21 trading days/month
                price_1m_ago = close_prices.iloc[-21]
                return_1m = ((current_price - price_1m_ago) / price_1m_ago) * 100
            else:
                return_1m = None
            
            # 3 Month Return
            if len(close_prices) >= 63:
                price_3m_ago = close_prices.iloc[-63]
                return_3m = ((current_price - price_3m_ago) / price_3m_ago) * 100
            else:
                return_3m = None
            
            # 6 Month Return
            if len(close_prices) >= 126:
                price_6m_ago = close_prices.iloc[-126]
                return_6m = ((current_price - price_6m_ago) / price_6m_ago) * 100
            else:
                return_6m = None
            
            # 4. LTP / 1Y High ratio
            ltp_to_1y_high_ratio = current_price / year_high
            
            features[symbol] = {
                'pe_ratio': pe_ratio,
                'discount_1y_high': discount_1y_high,
                'return_1m': return_1m,
                'return_3m': return_3m,
                'return_6m': return_6m,
                'ltp_to_1y_high_ratio': ltp_to_1y_high_ratio,
                'current_price': current_price,
                'year_high': year_high,
            }
            
        except Exception as e:
            # Skip stocks with errors
            continue
    
    return features


def scale_features(features_dict, selected_features):
    """
    Normalize features using StandardScaler.
    
    Args:
        features_dict: Dictionary from extract_features()
        selected_features: List of feature names to use
    
    Returns:
        Tuple: (scaled_features_array, valid_symbols, scaler)
    """
    # Build feature matrix
    valid_symbols = []
    feature_matrix = []
    
    for symbol, feature_data in features_dict.items():
        row = []
        has_valid_data = True
        
        for feat in selected_features:
            val = feature_data.get(feat)
            if val is None or not np.isfinite(val):
                has_valid_data = False
                break
            row.append(val)
        
        if has_valid_data:
            valid_symbols.append(symbol)
            feature_matrix.append(row)
    
    if len(valid_symbols) < 2:
        return None, valid_symbols, None
    
    # Convert to numpy array
    feature_array = np.array(feature_matrix)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_array)
    
    return scaled_features, valid_symbols, scaler


def run_clustering(scaled_features, k_value):
    """
    Apply KMeans clustering.
    
    Args:
        scaled_features: Normalized feature array from scale_features()
        k_value: Number of clusters (2-6)
    
    Returns:
        Array of cluster labels
    """
    # Ensure k_value is within valid range
    k_value = max(2, min(6, k_value))
    
    # Handle case where fewer stocks than k
    n_samples = scaled_features.shape[0]
    if n_samples < k_value:
        k_value = n_samples if n_samples >= 2 else 2
    
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    return cluster_labels


def generate_pca_data(scaled_features):
    """
    Reduce features to 2D using PCA for visualization.
    
    Args:
        scaled_features: Normalized feature array
    
    Returns:
        Tuple: (pca_x_array, pca_y_array)
    """
    n_samples = scaled_features.shape[0]
    
    # If only 1 sample, return dummy coordinates
    if n_samples == 1:
        return np.array([0]), np.array([0])
    
    # If only 2 samples, can't use PCA with 2 components on 2 samples
    # Use simple 2D projection
    if n_samples == 2:
        return np.array([0, 1]), np.array([0, 0])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    return pca_result[:, 0], pca_result[:, 1]


def cluster_portfolio(stock_symbols, selected_features, k_value):
    """
    Main function to cluster portfolio stocks.
    
    Args:
        stock_symbols: List of stock symbols
        selected_features: List of feature names to use
        k_value: Number of clusters
    
    Returns:
        List of dicts with stock_name, cluster_label, pca_x, pca_y
    """
    # Extract features
    features_dict = extract_features(stock_symbols)
    
    if len(features_dict) < 2:
        return {
            'success': False,
            'message': 'Need at least 2 stocks with valid data for clustering',
            'data': []
        }
    
    # Scale features
    scaled_features, valid_symbols, scaler = scale_features(features_dict, selected_features)
    
    if scaled_features is None or len(valid_symbols) < 2:
        return {
            'success': False,
            'message': 'Not enough valid data for clustering',
            'data': []
        }
    
    # Run clustering
    cluster_labels = run_clustering(scaled_features, k_value)
    
    # Generate PCA coordinates
    pca_x, pca_y = generate_pca_data(scaled_features)
    
    # Build result
    result_data = []
    for i, symbol in enumerate(valid_symbols):
        # Get stock name from features_dict if available
        stock_name = symbol
        if symbol in features_dict:
            # Use the symbol as name (can be enhanced to store actual names)
            stock_name = symbol
        
        result_data.append({
            'stock_name': stock_name,
            'stock_symbol': symbol,
            'cluster_label': int(cluster_labels[i]),
            'pca_x': float(pca_x[i]),
            'pca_y': float(pca_y[i]),
        })
    
    return {
        'success': True,
        'message': f'Successfully clustered {len(valid_symbols)} stocks into {k_value} clusters',
        'data': result_data
    }

