# AI Smart Energy Consumption Predictor

An intelligent machine learning system for predicting electricity consumption patterns using historical data, weather information, and temporal features. This project helps utilities and businesses optimize energy distribution and identify consumption anomalies.

## Features

- **Time-Series Forecasting**: Predict energy consumption 24-48 hours ahead
- **Multi-Feature Analysis**: Incorporates weather data, time of day, day of week, and holidays
- **Anomaly Detection**: Identifies unusual consumption patterns
- **Model Ensemble**: Combines LSTM, XGBoost, and statistical models for robust predictions
- **Real-time Inference**: Deploy models for live energy consumption predictions
- **Performance Metrics**: Comprehensive evaluation with MAPE, RMSE, and R² scores

## Project Structure

```
ai-energy-predictor/
├── data/
│   ├── raw/                 # Raw energy consumption data
│   └── processed/           # Preprocessed datasets
├── models/
│   └── trained_models/      # Saved model weights
├── src/
│   ├── predictor.py         # Main prediction engine
│   ├── data_processor.py    # Data preprocessing utilities
│   ├── config.py            # Configuration settings
│   ├── utils.py             # Helper functions
│   └── evaluate.py          # Model evaluation
├── notebooks/
│   └── exploratory.ipynb    # EDA and model exploration
├── example_usage.py         # Example script
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-energy-predictor.git
cd ai-energy-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.predictor import EnergyPredictor
import pandas as pd

# Initialize predictor
predictor = EnergyPredictor(model_type='ensemble')

# Load your data
data = pd.read_csv('data/raw/energy_data.csv')

# Train the model
predictor.train(data, test_size=0.2, epochs=50)

# Make predictions
predictions = predictor.predict(hours_ahead=24)

# Evaluate performance
metrics = predictor.evaluate()
print(f"MAPE: {metrics['mape']:.2f}%")
```

### Advanced Configuration

Edit `src/config.py` to customize:
- Model hyperparameters
- Feature engineering settings
- Training parameters
- Data preprocessing options

## Data Format

Expected CSV format with columns:
```
timestamp,consumption_kwh,temperature,humidity,pressure,hour,day_of_week,is_holiday
2024-01-01 00:00:00,1245.5,12.3,65,1013.25,0,0,0
2024-01-01 01:00:00,1180.2,11.8,68,1013.15,1,0,0
...
```

## Models Included

1. **LSTM Neural Network**
   - Captures temporal dependencies
   - Best for long-term patterns

2. **XGBoost Regressor**
   - Handles non-linear relationships
   - Fast inference time

3. **Statistical Models (ARIMA/ExponentialSmoothing)**
   - Baseline predictions
   - Interpretable results

4. **Ensemble Model**
   - Combines all models with weighted averaging
   - Best overall performance

## Performance Metrics

- **MAPE** (Mean Absolute Percentage Error): Measures average prediction accuracy
- **RMSE** (Root Mean Square Error): Penalizes larger errors
- **R² Score**: Proportion of variance explained

Typical performance on test data:
- MAPE: 5-8%
- RMSE: 150-250 kWh
- R²: 0.92-0.96

## API Endpoints (Optional)

For deployment, the project includes Flask API integration:

```bash
python app.py
```

Available endpoints:
- `POST /predict` - Get energy predictions
- `GET /metrics` - View model performance
- `GET /anomalies` - Detect consumption anomalies

## Configuration Options

Key settings in `src/config.py`:

```python
CONFIG = {
    'model_type': 'ensemble',  # 'lstm', 'xgboost', 'statistical', 'ensemble'
    'forecast_hours': 24,
    'train_test_split': 0.2,
    'feature_engineering': True,
    'normalize_data': True,
    'seasonal_decomposition': True,
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example notebooks

## Acknowledgments

- Built with TensorFlow/Keras, XGBoost, and scikit-learn
- Inspired by industry energy consumption forecasting standards
- Community contributions and feedback

## Roadmap

- [ ] Distributed training support
- [ ] Real-time data ingestion pipelines
- [ ] Mobile app for consumption insights
- [ ] Advanced anomaly detection algorithms
- [ ] Multi-region forecasting
- [ ] IoT device integration

---


