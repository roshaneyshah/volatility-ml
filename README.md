# Forecasting Models

Time series forecasting library for volatility and price prediction.

## Features
- GARCH volatility forecasting
- EWMA volatility forecasting
- XGBoost and LightGBM ML models
- LSTM neural network forecasting
- Prophet time series forecasting
- ARIMA price forecasting
- Ensemble forecasting

## Key finding
Walk-forward validation revealed that every point-forecast model in this library was systematically overconfident when evaluated on calibration rather than accuracy alone. Refitting on expanding windows and checking calibration against held-out volatility regimes consistently exposed overconfidence that accuracy metrics masked.

## Installation
pip install forecasting-models

## License
MIT License
