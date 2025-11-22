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

warnings.filterwarnings("ignore")

# =====================================================
# 1. DATA LOADING
# =====================================================

def load_sentiment_data(data_directory):
    """Combine yearly CSV files into one DataFrame."""
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
    data_directory = "/Users/user/Desktop/Urban Mood Analysis/dataverse_files/Sentiment Data - Country"
    combined_df = load_sentiment_data(data_directory)
    print(f"✅ Loaded {len(combined_df):,} rows from {combined_df['country'].nunique()} countries.\n")

    while True:
        print("\n🌍 What would you like to do?")
        print("1️⃣  Forecast sentiment for a country")
        print("2️⃣  Find the best season to visit a chosen country")
        print("3️⃣  Find the happiest countries in a chosen season")
        print("4️⃣  Exit")

        choice = input("Select an option (1-4): ").strip()

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
            print("👋 Exiting. See you next time!")
            break

        else:
            print("❌ Invalid option. Please choose 1-4.")




if __name__ == "__main__":
    main()
