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
import warnings
import pycountry
import pycountry_convert as pc

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
# MODEL EVALUATION FUNCTIONS
# ===============================

def evaluate_models(train, test):
    """
    Evaluate multiple time series models and return RMSE for each.
    
    Args:
        train: Training data (pandas Series)
        test: Test data (pandas Series)
        
    Returns:
        tuple: (results dictionary with model names and RMSEs, best model name)
    """
    results = {}
    
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

