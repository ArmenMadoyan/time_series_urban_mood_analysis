# Urban Mood Forecaster — Project Overview

A comprehensive time-series analysis and forecasting project that models global and country-level sentiment trends using both classical statistical methods and deep learning architectures.

The system includes data preprocessing, model training, evaluation, and visualization, delivered through a Streamlit web interface, a CLI tool, and a Jupyter notebook.

---

## Project Goals

- Build a complete **time-series forecasting pipeline** for global sentiment data.
- Compare classical models (AR, MA, ARIMA, SARIMA) with deep learning RNNs (LSTM, GRU, Bidirectional LSTM).
- Extract seasonal patterns to determine:
  - the **best season to visit each country**,
  - the **happiest countries by season**.
- Develop an **interactive application** for forecasting and exploring global emotional trends.
- Provide clear model comparison using RMSE, MAE, and visualization.

---

## Project Description

The Urban Mood Forecaster constructs weekly and daily sentiment time series (2012–2023), performs stationarity analysis, fits statistical and neural models, and generates forecast outputs.

The project provides three usage modes:

1. **Streamlit Web App** — interactive dashboard for forecasting and seasonal insights.
2. **CLI Tool** — fast terminal-based analysis.
3. **Jupyter Notebook** — full methodology, EDA, model comparison, and evaluation.

---

## Dataset

- Global/country-level daily sentiment scores  
- Time span: **2012–2023**  
- Source: processed Dataverse CSV files  
- Used for trend extraction, decomposition, and forecasting  
- Optional downsampling for faster experimentation

---

## Research Methods

- Exploratory data analysis and visualization  
- Stationarity checks (ADF, KPSS, seasonal plots)  
- Classical forecasting:
  - AR, MA, ARIMA, SARIMA
- Deep learning models:
  - LSTM, GRU, Bidirectional LSTM, Stacked LSTM
- Evaluation metrics: RMSE, MAE  
- Visual forecasting comparison  
- Streamlit UI design

---

## Methodology

1. Data extraction and cleaning  
2. Time-series construction  
3. Stationarity testing and seasonal differencing  
4. Model training (statistical + RNN)  
5. Forecast generation and evaluation  
6. Seasonal mood analysis  
7. Streamlit/CLI development  
8. Visual result interpretation  

---

## Data Pipeline

- Load raw CSV files  
- Clean, sort, and align timestamps  
- Normalize where necessary  
- Create sliding windows for RNNs  
- Train SARIMA and RNN models  
- Generate forecasts and seasonal summaries  
- Send results to UI/CLI

---

## Results

- **Bidirectional LSTM and LSTM** achieved the lowest RMSE and best visual tracking of sentiment.
- **SARIMA (1,0,1)(0,1,1)\_7** performed strongly as a classical baseline.
- **GRU** produced smoother predictions, under-reacting to sudden changes.
- **Stacked LSTM** overfit due to univariate input and model complexity.
- Seasonal analysis identified clear best-season patterns for most countries.
- Application outputs accurate forecasts and intuitive global visualizations.

---

## Project Structure

- Streamlit web interface  
- Backend model and forecasting engine  
- CLI terminal tool  
- Full research notebook  
- Dataset sampling utilities  
- Configurable seasonal and forecasting modules

---

## Outcomes and Conclusion

The project demonstrates that global sentiment can be forecasted effectively using hybrid statistical–deep learning pipelines.  
LSTM-based models capture short-term fluctuations and non-linearities, while SARIMA provides strong interpretability and seasonal modeling.

The final system integrates forecasting, seasonal recommendations, and visual analytics into a practical and accessible application.

---

## Slide Mapping for Presentation

- **Slide 1** — Project Title  
- **Slides 2–3** — Problem Definition  
- **Slides 4–5** — Dataset Details & Research Methods  
- **Slides 6–9** — Methodology  
- **Slides 10–13** — Data Pipeline & Time-Series Construction  
- **Slides 14–17** — Results  
- **Slides 18–19** — Project Structure, Outcomes, Conclusion  
- **Slide 20** — References  

**Note:** Some slides appear longer than listed due to animation (animation = +1 slide).

<<<<<<< HEAD
=======
## Models Available

### Traditional Models
- **AR(1)**: Autoregressive model
- **MA(1)**: Moving Average model
- **ARIMA(1,0,1)**: Combined AR and MA
- **SARIMA(1,0,1)(0,1,1,7)**: Seasonal ARIMA with weekly patterns

### Deep Learning Models (Optional)
- **LSTM**: Long Short-Term Memory network
- **GRU**: Gated Recurrent Unit
- **Bidirectional LSTM**: Processes sequences in both directions

## Troubleshooting

### TensorFlow Installation Issues

If TensorFlow fails to install:
1. The app works without it (traditional models only)
2. Try: `pip install tensorflow --upgrade`
3. For Apple Silicon (M1/M2): `pip install tensorflow-macos`

### Data Not Found

Ensure the `dataverse_files/` directory exists and contains CSV files.

### Port Already in Use

If port 8501 is busy:
```bash
streamlit run app.py --server.port 8502
```

## Requirements

See `requirments.txt` for full dependency list. Key packages:
- pandas, numpy
- plotly
- statsmodels
- scikit-learn
- streamlit
- tensorflow (optional, for RNN models)

## Canva Presentation link

(https://www.canva.com/design/DAG5h_aee48/r0NQOyU0gaLTIl1SwrZAPQ/edit?utm_content=DAG5h_aee48&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
>>>>>>> da4ab0c947d01504f8bf423aa397509cb153d5f1
