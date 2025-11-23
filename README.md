# Urban Mood Forecaster

A time series analysis and forecasting application for global sentiment trends using traditional statistical models and deep learning RNN models.

## Features

- **Sentiment Forecasting**: Predict future sentiment scores for any country using multiple models (AR, MA, ARIMA, SARIMA, LSTM, GRU, Bidirectional LSTM)
- **Seasonal Analysis**: Find the best season to visit a country based on historical sentiment
- **Travel Recommendations**: Discover the happiest countries by season
- **Interactive Visualizations**: Beautiful Plotly charts and maps

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/amadoyan/time_series_urban_mood_analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirments.txt
   ```

   **Note:** TensorFlow installation may take a few minutes. If you encounter issues or don't need RNN models, you can skip TensorFlow by commenting it out in `requirments.txt`.

## Running the Application

### Option 1: Streamlit Web App (Recommended)

Run the main web application:

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

**Features:**
- Interactive web interface
- Three main functions: Forecast, Best Season, Happiest Countries
- Optional RNN models (LSTM, GRU, Bidirectional LSTM)
- Beautiful visualizations

### Option 2: Command Line Interface

Run the CLI version:

```bash
python User_Input.py
```

**Features:**
- Terminal-based menu system
- Same functionality as web app
- Good for quick analysis without a browser

### Option 3: Jupyter Notebook

Open and run the analysis notebook:

```bash
jupyter notebook "Urban Mood Analysis.ipynb"
```

**Features:**
- Comprehensive exploratory data analysis
- Model comparison and evaluation
- Deep learning model implementations
- Interactive visualizations

## Usage Guide

### Web App Usage

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Forecast Sentiment:**
   - Click "Forecast Sentiment" button
   - Select a country from the dropdown
   - Choose forecast horizon (7-365 days)
   - Optionally enable RNN models
   - Click "Run Forecast"
   - View model comparison and forecast chart

3. **Best Season to Visit:**
   - Click "Best Season to Visit" button
   - Select a country
   - View seasonal sentiment breakdown
   - See recommended season

4. **Happiest Countries by Season:**
   - Click "Happiest Countries by Season" button
   - Select a season (winter, spring, summer, autumn)
   - View top 15 happiest countries
   - Optionally view continent breakdown

### RNN Models

RNN models (LSTM, GRU, Bidirectional LSTM) are optional and require TensorFlow:

- **To use RNN models:** Check the "Include RNN Models" checkbox
- **Training time:** RNN models take longer to train (30-60 seconds)
- **Better accuracy:** Often provide better forecasts for complex patterns
- **Requirements:** TensorFlow must be installed

If TensorFlow is not installed, the app will work with traditional models only.

## Project Structure

```
time_series_urban_mood_analysis/
├── app.py                    # Streamlit web application (Frontend)
├── backend.py                # Business logic and models (Backend)
├── User_Input.py             # CLI version
├── Urban Mood Analysis.ipynb # Jupyter notebook analysis
├── requirments.txt           # Python dependencies
├── create_sampled_dataset.py # Utility to downsample datasets safely
├── dataverse_files/          # Data directory (CSV files)
│   ├── Sentiment Data - Country/
│   ├── Sentiment Data - State/
│   ├── Sentiment Data - County/
│   └── Sentiment Data - World/
├── Background.png            # App background image
└── cityscape font.ttf        # Custom font
```

## Data

The application uses sentiment data from CSV files in the `dataverse_files/` directory:
- **Country-level data**: Used by default in the app
- **Time range**: 2012-2023
- **Format**: Daily sentiment scores per country

## Working with Smaller Samples

The raw `dataverse_files/` directory can be quite large. A full copy now lives in `dataverse_files_full/` so the original download stays untouched. Use the new `create_sampled_dataset.py` utility whenever you need a lighter-weight dataset:

```bash
# Keep only 25% of each CSV in a fresh directory
python3 create_sampled_dataset.py \
  --source dataverse_files_full \
  --destination dataverse_files_sampled \
  --fraction 0.25
```

Point training scripts to the sampled directory, or let the app/CLI automatically sample rows by setting an environment variable before launching:

```bash
export SENTIMENT_SAMPLE_FRACTION=0.25
streamlit run app.py
# or
SENTIMENT_SAMPLE_FRACTION=0.25 python User_Input.py
```

Both approaches leave the original dataset untouched while allowing faster iterations.

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

## License

This project is for educational and research purposes.

## Support

For issues or questions, check the code comments or notebook documentation.
