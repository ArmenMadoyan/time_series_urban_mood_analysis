"""
Urban Mood Forecasting & Travel Recommendation System
-----------------------------------------------------
Analyze and forecast sentiment trends across countries, 
compare time-series models (AR, MA, ARIMA, SARIMA), 
and recommend the best travel seasons or destinations 
based on predicted mood levels.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
import difflib
import difflib
import pycountry
import pycountry_convert as pc
from backend import forecast_with_rnn, TENSORFLOW_AVAILABLE

warnings.filterwarnings("ignore")


def _resolve_sample_fraction():
    """Read SENTIMENT_SAMPLE_FRACTION env var and keep it in (0, 1]."""
    env_value = os.environ.get("SENTIMENT_SAMPLE_FRACTION")
    if not env_value:
        return 1.0
    try:
        fraction = float(env_value)
    except ValueError:
        print(f"[WARN] Invalid SENTIMENT_SAMPLE_FRACTION='{env_value}'. Using full dataset.")
        return 1.0
    if not 0 < fraction <= 1:
        print(f"[WARN] SENTIMENT_SAMPLE_FRACTION must be between 0 and 1. Got {fraction}. Using full dataset.")
        return 1.0
    return fraction

# =====================================================
# 1. DATA LOADING
# =====================================================

def load_sentiment_data(data_directory, sample_fraction=1.0, random_state=42):
    """Combine yearly CSV files into one DataFrame."""
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("sample_fraction must be in the (0, 1] interval.")

    df_list = []
    csv_files = [f for f in os.listdir(data_directory) if f.endswith(".csv")]

    for file in csv_files:
        file_path = os.path.join(data_directory, file)
        temp_df = pd.read_csv(file_path)
        year = file.split("_")[-1].replace(".csv", "").replace("-1", "")
        temp_df["year"] = int(year)

        if 0 < sample_fraction < 1:
            temp_df = temp_df.sample(frac=sample_fraction, random_state=random_state)

        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.rename(columns={"NAME_0": "country"})
    combined_df["DATE"] = pd.to_datetime(combined_df["DATE"])
    combined_df = combined_df.sort_values(["country", "DATE"])
    return combined_df


# =====================================================
# 2. COUNTRY FILTERING
# =====================================================

def get_country_data(combined_df, country):
    """Return interpolated daily sentiment series for a given country."""
    df = combined_df[combined_df["country"] == country][["DATE", "SCORE"]]
    if df.empty:
        raise ValueError(f"No data found for '{country}'.")
    df = df.set_index("DATE").asfreq("D").interpolate("linear")
    return df


# =====================================================
# 3. MODELING FUNCTIONS
# =====================================================

def evaluate_models(train, test):
    """Train AR, MA, ARIMA, SARIMA models and compare RMSE."""
    results = {}

    # AR
    ar = ARIMA(train, order=(1, 0, 0)).fit()
    results["AR(1)"] = np.sqrt(mean_squared_error(test, ar.forecast(len(test))))

    # MA
    ma = ARIMA(train, order=(0, 0, 1)).fit()
    results["MA(1)"] = np.sqrt(mean_squared_error(test, ma.forecast(len(test))))

    # ARIMA
    arima = ARIMA(train, order=(1, 0, 1)).fit()
    results["ARIMA(1,0,1)"] = np.sqrt(mean_squared_error(test, arima.forecast(len(test))))

    # SARIMA (best config from your tuning)
    sarima = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
    results["SARIMA(1,0,1)(0,1,1,7)"] = np.sqrt(mean_squared_error(test, sarima.forecast(len(test))))

    best_model = min(results, key=results.get)
    return results, best_model


def forecast_future(df, steps=90):
    """Forecast future sentiment using SARIMA."""
    model = SARIMAX(df["SCORE"], order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)).fit(disp=False)
    forecast = model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return future_dates, pred_mean, pred_ci


# =====================================================
# 4. VISUALIZATION
# =====================================================

def plot_forecast(df, future_dates, pred_mean, pred_ci, country):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["SCORE"], mode="lines", name="Observed"))
    fig.add_trace(go.Scatter(x=future_dates, y=pred_mean, mode="lines", name="Forecast", line=dict(color="purple")))
    fig.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates[::-1].tolist(),
        y=pred_ci.iloc[:, 0].tolist() + pred_ci.iloc[:, 1][::-1].tolist(),
        fill="toself",
        fillcolor="rgba(200,150,255,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval"
    ))
    fig.update_layout(title=f"Forecasted Sentiment for {country}", template="plotly_white")
    fig.show()


# =====================================================
# 5. TRAVEL RECOMMENDER
# =====================================================

def recommend_by_season(combined_df, season):
    """Recommend happiest countries for a given season."""
    season_months = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    df = combined_df.copy()
    df["month"] = df["DATE"].dt.month
    top = (
        df[df["month"].isin(season_months[season.lower()])]
        .groupby("country")["SCORE"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"\n🌤 Top 10 happiest countries in {season.capitalize()}:")
    print(top)
    return top


# =====================================================
# 6. MAIN EXECUTION
# =====================================================


def _prompt_int(prompt, default, min_value=None, max_value=None):
    """Prompt user for an integer; enforce bounds, fall back to default on invalid input."""
    raw = input(f"{prompt} [default={default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print("⚠️ Invalid number. Using default.")
        return default
    if min_value is not None and value < min_value:
        print(f"⚠️ Value must be at least {min_value}. Using default.")
        return default
    if max_value is not None and value > max_value:
        print(f"⚠️ Value must be at most {max_value}. Using default.")
        return default
    return value


def run_rnn_forecast(combined_df):
    """Interactive workflow for training/using an RNN model."""
    if not TENSORFLOW_AVAILABLE:
        print("\n❌ TensorFlow is not installed, so RNN models are unavailable.")
        print("   Install TensorFlow (see requirments.txt) and try again.\n")
        return

    available_countries = sorted(combined_df["country"].unique())
    country = input("\nEnter a country for the RNN forecast (or 'back' to menu): ").strip()
    if country.lower() == "back":
        return

    match = [c for c in available_countries if c.lower() == country.lower()]
    if not match:
        suggestion = difflib.get_close_matches(country, available_countries, n=3, cutoff=0.6)
        if suggestion:
            print(f"\n⚠️ Couldn't find '{country}'. Possible matches: {', '.join(suggestion)}")
        else:
            print("❌ No data found for that country.")
        return

    country = match[0]
    df = get_country_data(combined_df, country)

    print("\nAvailable RNN architectures:")
    model_options = {
        "1": ("lstm", "LSTM"),
        "2": ("gru", "GRU"),
        "3": ("bilstm", "Bidirectional LSTM"),
        "4": ("stacked_lstm", "Stacked LSTM")
    }
    for key, (_, label) in model_options.items():
        print(f"{key}. {label}")
    model_choice = input("Choose a model type (1-4): ").strip()
    model_type, model_label = model_options.get(model_choice, ("lstm", "LSTM"))

    seq_length = _prompt_int("Sequence length (timesteps fed into the network)", 30, min_value=10, max_value=180)
    steps = _prompt_int("Forecast horizon (days ahead)", 90, min_value=7, max_value=365)
    epochs = _prompt_int("Max training epochs (used when no saved model exists yet)", 20, min_value=5, max_value=100)

    print(f"\n🤖 Training/using {model_label} model for {country}...")
    future_dates, pred_mean, pred_ci = forecast_with_rnn(
        df, model_type=model_type, seq_length=seq_length, steps=steps, epochs=epochs, country=country
    )

    if pred_mean is None:
        print("❌ RNN forecast failed. Try different parameters or fall back to classical models.")
        return

    print(f"✅ Generated {steps}-day {model_label} forecast for {country}.")
    plot_forecast(df, future_dates, pred_mean, pred_ci, f"{country} ({model_label})")


# =====================================================
# Helper: Map ISO3 → Continent
# =====================================================

def get_continent_from_iso3(iso3):
    """Return continent name given ISO3 country code."""
    try:
        country_alpha2 = pc.country_alpha3_to_country_alpha2(iso3)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continents = {
            "AF": "Africa",
            "NA": "North America",
            "OC": "Oceania",
            "AN": "Antarctica",
            "AS": "Asia",
            "EU": "Europe",
            "SA": "South America"
        }
        return continents.get(continent_code, "Unknown")
    except Exception:
        return "Unknown"

# =====================================================
# 7. Menu System
# =====================================================

def main():
    # Resolve dataset path relative to this script so it works on any machine
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(base_dir, "dataverse_files", "Sentiment Data - Country")
    sample_fraction = _resolve_sample_fraction()
    combined_df = load_sentiment_data(data_directory, sample_fraction=sample_fraction)
    print(f"✅ Loaded {len(combined_df):,} rows from {combined_df['country'].nunique()} countries.")
    if sample_fraction < 1.0:
        print(f"⚡ Using {sample_fraction:.0%} of the dataset for faster experimentation.")
    print()

    while True:
        print("\n🌍 What would you like to do?")
        print("1️⃣  Forecast sentiment for a country")
        print("2️⃣  Find the best season to visit a chosen country")
        print("3️⃣  Find the happiest countries in a chosen season")
        print("4️⃣  Run an RNN-based forecast")
        print("5️⃣  Exit")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            # === FORECAST ===
            available_countries = sorted(combined_df["country"].unique())
            while True:
                country = input("\nEnter a country to forecast (or 'back' to menu): ").strip()
                if country.lower() == "back":
                    break

                match = [c for c in available_countries if c.lower() == country.lower()]
                if match:
                    country = match[0]
                    print(f"📈 Using country: {country}\n")
                    df = get_country_data(combined_df, country)

                    # Train/test split
                    train_size = int(len(df) * 0.8)
                    train, test = df["SCORE"][:train_size], df["SCORE"][train_size:]

                    results, best_model = evaluate_models(train, test)
                    print("\n📊 Model RMSEs:")
                    for model, rmse in results.items():
                        print(f"  {model}: {rmse:.5f}")
                    print(f"\n🏆 Best Model for {country}: {best_model}")

                    # Forecast future
                    future_dates, pred_mean, pred_ci = forecast_future(df)
                    plot_forecast(df, future_dates, pred_mean, pred_ci, country)
                    break

                suggestion = difflib.get_close_matches(country, available_countries, n=3, cutoff=0.6)
                if suggestion:
                    print(f"\n⚠️ Couldn't find '{country}'. Possible matches: {', '.join(suggestion)}")
                    continue
                else:
                    print("❌ No match found. Please retype.")
                    continue

        elif choice == "2":
            # === BEST SEASON TO TRAVEL TO COUNTRY ===
            country = input("\nEnter a country name: ").strip()
            df = get_country_data(combined_df, country)
            df["month"] = df.index.month
            monthly_avg = df.groupby("month")["SCORE"].mean()

            best_month = monthly_avg.idxmax()
            best_score = monthly_avg.max()

            print(f"\n🌞 The happiest month for {country} is {best_month} with avg sentiment {best_score:.3f}.\n")
            fig = px.bar(
                x=monthly_avg.index,
                y=monthly_avg.values,
                labels={"x": "Month", "y": "Average Sentiment"},
                title=f"Average Sentiment by Month — {country}",
                template="plotly_white",
            )
            fig.add_vline(x=best_month, line_dash="dash", line_color="green")
            fig.show()

        elif choice == "3":
            # === BEST COUNTRIES FOR A SEASON ===
            season = input("Enter a season (winter/spring/summer/autumn): ").strip().lower()
            top = recommend_by_season(combined_df, season)

            # --- Optional continent filter ---
            filt = input("Filter by continent? (y/n): ").lower()
            if filt == "y":
                top_countries = top.index.tolist()
                continent_map = {}
                for country in top_countries:
                    try:
                        iso3 = pycountry.countries.lookup(country).alpha_3
                        continent_map[country] = get_continent_from_iso3(iso3)
                    except:
                        continent_map[country] = "Unknown"

                print("\n🌐 Continent breakdown:")
                for c, cont in continent_map.items():
                    print(f"{c}: {cont}")

        elif choice == "4":
            run_rnn_forecast(combined_df)

        elif choice == "5":
            print("👋 Exiting. See you next time!")
            break

        else:
            print("❌ Invalid option. Please choose 1-5.")




if __name__ == "__main__":
    main()
