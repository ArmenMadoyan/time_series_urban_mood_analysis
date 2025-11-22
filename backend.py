"""
Backend functions for Urban Mood Forecaster
Handles data processing, model training, forecasting, and recommendations
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import pycountry
import pycountry_convert as pc

# Try to import TensorFlow for RNN models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings("ignore")


# ===============================
# DATA LOADING FUNCTIONS
# ===============================

def load_sentiment_data(data_directory):
    """
    Load and combine all CSV files from the data directory.
    
    Args:
        data_directory: Path to directory containing CSV files
        
    Returns:
        Combined DataFrame with all sentiment data
    """
    df_list = []
    csv_files = [f for f in os.listdir(data_directory) if f.endswith(".csv")]
    
    for file in csv_files:
        file_path = os.path.join(data_directory, file)
        temp_df = pd.read_csv(file_path)
        year = file.split("_")[-1].replace(".csv", "").replace("-1", "")
        temp_df["year"] = int(year)
        df_list.append(temp_df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.rename(columns={"NAME_0": "country"})
    combined_df["DATE"] = pd.to_datetime(combined_df["DATE"])
    combined_df = combined_df.sort_values(["country", "DATE"])
    
    return combined_df


def get_country_data(combined_df, country):
    """
    Extract and prepare time series data for a specific country.
    
    Args:
        combined_df: Combined DataFrame with all sentiment data
        country: Country name to filter
        
    Returns:
        DataFrame with DATE as index and SCORE column, interpolated to daily frequency
    """
    df = combined_df[combined_df["country"] == country][["DATE", "SCORE"]]
    df = df.set_index("DATE").asfreq("D").interpolate("linear")
    return df


# ===============================
# RNN PREPROCESSING FUNCTIONS
# ===============================

def create_sequences(data, seq_length):
    """Create sequences for RNN input"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def prepare_rnn_data(train_data, test_data, seq_length=30, scaler=None):
    """Prepare data for RNN models with scaling"""
    # Combine train and test for scaling
    all_data = np.concatenate([train_data.values, test_data.values]).reshape(-1, 1)
    
    # Fit scaler on all data (or use provided scaler)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(all_data)
    else:
        scaled_data = scaler.transform(all_data)
    
    # Split back
    train_scaled = scaled_data[:len(train_data)]
    test_scaled = scaled_data[len(train_data):]
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled.flatten(), seq_length)
    X_test, y_test = create_sequences(test_scaled.flatten(), seq_length)
    
    # Reshape for RNN: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler


# ===============================
# RNN MODEL FUNCTIONS
# ===============================

def train_lstm_model(X_train, y_train, X_test, y_test, scaler, seq_length=30, epochs=50, verbose=0):
    """Train LSTM model and return predictions and RMSE"""
    if not TENSORFLOW_AVAILABLE:
        return None, None
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Build LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=0)
    
    # Train model
    lstm_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    # Make predictions
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
    
    # Calculate RMSE
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred_scaled))
    
    return lstm_pred, lstm_rmse


def train_gru_model(X_train, y_train, X_test, y_test, scaler, seq_length=30, epochs=50, verbose=0):
    """Train GRU model and return predictions and RMSE"""
    if not TENSORFLOW_AVAILABLE:
        return None, None
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Build GRU model
    gru_model = Sequential([
        GRU(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        GRU(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=0)
    
    # Train model
    gru_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    # Make predictions
    gru_pred_scaled = gru_model.predict(X_test, verbose=0)
    gru_pred = scaler.inverse_transform(gru_pred_scaled).flatten()
    
    # Calculate RMSE
    gru_rmse = np.sqrt(mean_squared_error(y_test, gru_pred_scaled))
    
    return gru_pred, gru_rmse


def train_bilstm_model(X_train, y_train, X_test, y_test, scaler, seq_length=30, epochs=50, verbose=0):
    """Train Bidirectional LSTM model and return predictions and RMSE"""
    if not TENSORFLOW_AVAILABLE:
        return None, None
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Build Bidirectional LSTM model
    bilstm_model = Sequential([
        Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=(seq_length, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(50, activation='relu', return_sequences=False)),
        Dropout(0.2),
        Dense(1)
    ])
    
    bilstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=0)
    
    # Train model
    bilstm_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    # Make predictions
    bilstm_pred_scaled = bilstm_model.predict(X_test, verbose=0)
    bilstm_pred = scaler.inverse_transform(bilstm_pred_scaled).flatten()
    
    # Calculate RMSE
    bilstm_rmse = np.sqrt(mean_squared_error(y_test, bilstm_pred_scaled))
    
    return bilstm_pred, bilstm_rmse


# ===============================
# MODEL EVALUATION FUNCTIONS
# ===============================

def evaluate_models(train, test, include_rnn=True, seq_length=30, rnn_epochs=30):
    """
    Evaluate multiple time series models and return RMSE for each.
    Includes traditional models and optional RNN models.
    
    Args:
        train: Training data (pandas Series)
        test: Test data (pandas Series)
        include_rnn: Whether to include RNN models (requires TensorFlow)
        seq_length: Sequence length for RNN models
        rnn_epochs: Number of epochs for RNN training
        
    Returns:
        tuple: (results dictionary with model names and RMSEs, best model name)
    """
    results = {}
    
    # Traditional Models
    # AR(1) Model
    ar = ARIMA(train, order=(1, 0, 0)).fit()
    results["AR(1)"] = np.sqrt(mean_squared_error(test, ar.forecast(len(test))))
    
    # MA(1) Model
    ma = ARIMA(train, order=(0, 0, 1)).fit()
    results["MA(1)"] = np.sqrt(mean_squared_error(test, ma.forecast(len(test))))
    
    # ARIMA(1,0,1) Model
    arima = ARIMA(train, order=(1, 0, 1)).fit()
    results["ARIMA(1,0,1)"] = np.sqrt(mean_squared_error(test, arima.forecast(len(test))))
    
    # SARIMA(1,0,1)(0,1,1,7) Model
    sarima = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
    results["SARIMA(1,0,1)(0,1,1,7)"] = np.sqrt(mean_squared_error(test, sarima.forecast(len(test))))
    
    # RNN Models (if TensorFlow is available and requested)
    if include_rnn and TENSORFLOW_AVAILABLE and len(train) > seq_length and len(test) > seq_length:
        try:
            # Prepare RNN data
            X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn, scaler_rnn = prepare_rnn_data(
                train, test, seq_length=seq_length
            )
            
            # LSTM Model
            lstm_pred, lstm_rmse = train_lstm_model(
                X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn, 
                scaler_rnn, seq_length=seq_length, epochs=rnn_epochs, verbose=0
            )
            if lstm_rmse is not None:
                # Calculate RMSE on original scale
                test_aligned = test.iloc[seq_length:]
                lstm_rmse_original = np.sqrt(mean_squared_error(test_aligned, lstm_pred))
                results["LSTM"] = lstm_rmse_original
            
            # GRU Model
            gru_pred, gru_rmse = train_gru_model(
                X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn,
                scaler_rnn, seq_length=seq_length, epochs=rnn_epochs, verbose=0
            )
            if gru_rmse is not None:
                test_aligned = test.iloc[seq_length:]
                gru_rmse_original = np.sqrt(mean_squared_error(test_aligned, gru_pred))
                results["GRU"] = gru_rmse_original
            
            # Bidirectional LSTM Model
            bilstm_pred, bilstm_rmse = train_bilstm_model(
                X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn,
                scaler_rnn, seq_length=seq_length, epochs=rnn_epochs, verbose=0
            )
            if bilstm_rmse is not None:
                test_aligned = test.iloc[seq_length:]
                bilstm_rmse_original = np.sqrt(mean_squared_error(test_aligned, bilstm_pred))
                results["Bidirectional LSTM"] = bilstm_rmse_original
                
        except Exception as e:
            # If RNN training fails, continue without RNN models
            pass
    
    best_model = min(results, key=results.get)
    return results, best_model


def forecast_future(df, steps=90):
    """
    Generate future sentiment forecast using SARIMA model.
    
    Args:
        df: DataFrame with DATE index and SCORE column
        steps: Number of days to forecast ahead
        
    Returns:
        tuple: (future_dates, predicted_mean, confidence_interval)
    """
    model = SARIMAX(df["SCORE"], order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
    forecast = model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    
    return future_dates, pred_mean, pred_ci


# ===============================
# RECOMMENDATION FUNCTIONS
# ===============================

def recommend_by_season(combined_df, season):
    """
    Recommend top happiest countries for a given season.
    
    Args:
        combined_df: Combined DataFrame with all sentiment data
        season: Season name (winter, spring, summer, autumn)
        
    Returns:
        Series with top 15 countries sorted by average sentiment
    """
    season_months = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11]
    }
    
    df = combined_df.copy()
    df["month"] = df["DATE"].dt.month
    
    top = (
        df[df["month"].isin(season_months[season.lower()])]
        .groupby("country")["SCORE"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    
    return top


def get_seasonal_scores(df):
    """
    Calculate average sentiment scores for each season.
    
    Args:
        df: DataFrame with country data including DATE and SCORE columns
        
    Returns:
        Dictionary with season names as keys and average sentiment as values
    """
    season_labels = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    
    df_copy = df.copy()
    df_copy["month"] = df_copy["DATE"].dt.month
    
    season_scores = {}
    for season, months in season_labels.items():
        season_scores[season] = df_copy[df_copy["month"].isin(months)]["SCORE"].mean()
    
    return season_scores


# ===============================
# GEOGRAPHY HELPER FUNCTIONS
# ===============================

def get_continent_from_country(country_name):
    """
    Get continent name for a given country using fuzzy matching.
    
    Args:
        country_name: Country name (string)
        
    Returns:
        Continent name or "Unknown" if not found
    """
    try:
        match = pycountry.countries.search_fuzzy(country_name)[0]
        alpha2 = match.alpha_2
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        continent_name = {
            "AF": "Africa",
            "NA": "North America",
            "OC": "Oceania",
            "AN": "Antarctica",
            "AS": "Asia",
            "EU": "Europe",
            "SA": "South America",
        }.get(continent_code, "Unknown")
        return continent_name
    except Exception:
        return "Unknown"


def get_continent_breakdown(country_list):
    """
    Get continent breakdown for a list of countries.
    
    Args:
        country_list: List of country names
        
    Returns:
        List of tuples (country, continent)
    """
    continent_data = []
    for country in country_list:
        continent_name = get_continent_from_country(country)
        continent_data.append((country, continent_name))
    return continent_data

