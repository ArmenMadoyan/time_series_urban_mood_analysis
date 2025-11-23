# app.py — Urban Mood Forecaster (Professional Edition)
# Streamlit UI Frontend

import os
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from backend import (
    load_sentiment_data,
    get_country_data,
    evaluate_models,
    forecast_future,
    recommend_by_season,
    get_seasonal_scores,
    get_continent_breakdown,
    TENSORFLOW_AVAILABLE
)

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="Urban Mood Forecaster", page_icon="🌍", layout="wide")


# === Custom Background & Font ===
def add_bg_and_font(bg_path, font_path):
    """Apply local background and custom font styling."""
    with open(bg_path, "rb") as f:
        bg_encoded = base64.b64encode(f.read()).decode()
    with open(font_path, "rb") as f:
        font_encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    @font-face {{
        font-family: 'Cityscape';
        src: url(data:font/ttf;base64,{font_encoded}) format('truetype');
    }}

    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bg_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(255, 255, 255, 0.65);
        z-index: 0;
    }}

    h1 {{
        font-family: 'Cityscape', serif !important;
        font-size: 120px !important;
        letter-spacing: 3px;
        text-align: center;
        color: #111 !important;
        margin-top: 80px;
        margin-bottom: 60px;
        text-transform: uppercase;
    }}

    h2, h3, label, p {{
        color: #111 !important;
    }}

    .stButton>button {{
        background: rgba(255, 255, 255, 0.9);
        color: #000 !important;
        font-size: 22px !important;
        font-weight: 600;
        border: 2px solid #000;
        border-radius: 50px;
        padding: 0.7rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}

    .stButton>button:hover {{
        background: #000;
        color: white !important;
        transform: scale(1.05);
    }}

    div[data-testid="stHorizontalBlock"] {{
        margin-top: 120px !important;
    }}

    .block-container {{
        background: rgba(255,255,255,0.8);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


# ✅ Apply local resources dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
add_bg_and_font(
    os.path.join(BASE_DIR, "Background.png"),
    os.path.join(BASE_DIR, "cityscape font.ttf")
)


# ===============================
# DATA LOADING WITH CACHING
# ===============================
@st.cache_data
def load_data_cached(data_directory):
    """Wrapper to cache data loading for Streamlit."""
    return load_sentiment_data(data_directory)


# ===============================
# MAIN INTERFACE
# ===============================
st.markdown("<h1>Urban Mood Forecaster</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:22px;'>Analyze global sentiment trends, discover the most positive travel seasons, and explore data-driven insights about city mood dynamics.</p>",
    unsafe_allow_html=True
)
st.write("")

data_directory = os.path.join(BASE_DIR, "dataverse_files", "Sentiment Data - Country")
combined_df = load_data_cached(data_directory)

# --- Persistent option storage ---
if "option" not in st.session_state:
    st.session_state.option = None

# --- Main navigation buttons ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Forecast Sentiment", use_container_width=True):
        st.session_state.option = "forecast"
with col2:
    if st.button("Best Season to Visit", use_container_width=True):
        st.session_state.option = "best_season"
with col3:
    if st.button("Happiest Countries by Season", use_container_width=True):
        st.session_state.option = "happiest"


# ===============================
# FUNCTIONAL SECTIONS
# ===============================

# ---------- FORECAST SECTION ----------
if st.session_state.option == "forecast":
    st.subheader("Forecast Sentiment for a Country")

    countries = sorted(combined_df["country"].unique())
    country = st.selectbox("Select a country:", countries)

    df = get_country_data(combined_df, country)
    st.line_chart(df["SCORE"], use_container_width=True)

    train_size = int(len(df) * 0.8)
    train, test = df["SCORE"][:train_size], df["SCORE"][train_size:]

    forecast_horizon = st.number_input(
        "How many days ahead would you like to forecast?",
        min_value=7, max_value=365, value=90, step=7,
        help="Choose between 7 and 365 days."
    )
    
    # RNN Model Options
    col1, col2 = st.columns(2)
    with col1:
        include_rnn = st.checkbox(
            "Include RNN Models (LSTM, GRU, Bidirectional LSTM)",
            value=False,
            help="Requires TensorFlow. May take longer to train but can provide better accuracy."
        )
    with col2:
        if include_rnn:
            if TENSORFLOW_AVAILABLE:
                st.success("✅ TensorFlow available - RNN models will be included")
            else:
                st.warning("⚠️ TensorFlow not installed - RNN models will be skipped")
                include_rnn = False

    if st.button("Run Forecast"):
        with st.spinner("Training models... This may take a moment, especially with RNN models."):
            results, best_model = evaluate_models(train, test, include_rnn=include_rnn, country=country)
        st.write("### Model RMSEs")
        st.dataframe(pd.DataFrame(list(results.items()), columns=["Model", "RMSE"]))
        st.success(f"Best Performing Model: {best_model}")

        # Determine which model to use for forecasting
        forecast_model_type = "sarima"  # Default
        if best_model == "LSTM":
            forecast_model_type = "lstm"
        elif best_model == "GRU":
            forecast_model_type = "gru"
        elif best_model == "Bidirectional LSTM":
            forecast_model_type = "bilstm"
        elif best_model == "Stacked LSTM":
            forecast_model_type = "stacked_lstm"
        
        # Generate forecast using best model (will load saved model if available)
        future_dates, pred_mean, pred_ci = forecast_future(
            df, 
            steps=int(forecast_horizon),
            model_type=forecast_model_type,
            country=country
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["SCORE"], mode="lines", name="Observed",
                                 line=dict(color="#2C2C2C", width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=pred_mean, mode="lines",
                                 name=f"{forecast_horizon}-Day Forecast",
                                 line=dict(color="#0055FF", width=3)))
        fig.add_trace(go.Scatter(x=future_dates, y=pred_ci.iloc[:, 0], fill=None, mode="lines",
                                 line=dict(color="rgba(0,136,255,0)", width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=pred_ci.iloc[:, 1], fill='tonexty', mode="lines",
                                 line=dict(color="rgba(0,136,255,0)", width=0),
                                 fillcolor="rgba(0,136,255,0.15)", name="Confidence Interval"))
        fig.update_layout(
            title=f"Forecasted Sentiment for {country} ({forecast_horizon} Days Ahead)",
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------- BEST SEASON TO VISIT ----------
elif st.session_state.option == "best_season":
    st.subheader("Best Season to Visit a Country")

    countries = sorted(combined_df["country"].unique())
    country = st.selectbox("Select a country:", countries)

    df = combined_df[combined_df["country"] == country].copy()
    season_scores = get_seasonal_scores(df)
    season_df = pd.DataFrame(list(season_scores.items()), columns=["Season", "Average Sentiment"]).sort_values(
        "Average Sentiment", ascending=False
    )

    best_season = season_df.iloc[0]["Season"].capitalize()

    st.write(f"### 🌤️ Sentiment by Season in {country}")
    st.dataframe(season_df)

    fig = px.bar(
        season_df,
        x="Season",
        y="Average Sentiment",
        color="Season",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"Seasonal Sentiment Averages — {country}",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✨ The best season to visit **{country}** is **{best_season}**, with the highest average sentiment score!")


# ---------- HAPPIEST COUNTRIES ----------
elif st.session_state.option == "happiest":
    st.subheader("Happiest Countries by Season")

    combined_df["country"] = combined_df["country"].astype(str).str.strip()
    season = st.selectbox("Choose a season:", ["winter", "spring", "summer", "autumn"])
    top = recommend_by_season(combined_df, season)

    if top.empty:
        st.warning("No data found for this season. Please check your dataset or date formatting.")
    else:
        st.dataframe(top.rename("Average Sentiment"))
        st.success(f"Top {len(top)} happiest countries in {season.capitalize()} 🌞")

        if st.checkbox("Show Continent Breakdown"):
            continent_data = get_continent_breakdown(top.index.tolist())

            continent_df = pd.DataFrame(continent_data, columns=["Country", "Continent"])
            st.markdown("### 🌍 Continent Breakdown")
            st.dataframe(continent_df)

            continent_counts = continent_df["Continent"].value_counts().reset_index()
            continent_counts.columns = ["Continent", "Count"]

            fig = px.bar(
                continent_counts,
                x="Continent",
                y="Count",
                text="Count",
                title=f"Distribution of Top {len(top)} Happiest Countries by Continent ({season.capitalize()})",
                color="Continent",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="Continent",
                yaxis_title="Number of Countries",
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
