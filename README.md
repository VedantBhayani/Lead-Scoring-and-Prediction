# Lead Scoring Prediction System

A comprehensive machine learning system for predicting lead scores using various features and advanced modeling techniques.

## Overview

This system provides an end-to-end solution for lead scoring, including:
- Data preprocessing and feature engineering
- Model training with multiple algorithms (XGBoost, LightGBM)
- Model evaluation and visualization
- Interactive dashboard for analysis
- Batch prediction capabilities

## Features

- **Data Preprocessing**: Automated data cleaning and transformation
- **Feature Engineering**: Advanced feature creation and selection
- **Model Training**: Support for multiple ML algorithms with hyperparameter optimization
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Interactive Dashboard**: Streamlit-based dashboard for analysis
- **Batch Prediction**: Support for processing multiple leads at once

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lead-scoring-prediction.git
cd lead-scoring-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Mode
```bash
python -m MAIN.src.main --mode train --data path/to/training_data.csv
```

### Prediction Mode
```bash
python -m MAIN.src.main --mode predict --data path/to/leads.csv --model path/to/model.joblib
```

### Dashboard Mode
```bash
python -m MAIN.src.main --mode dashboard
```

## Project Structure

```
lead-scoring-prediction/
├── MAIN/
│   ├── src/
│   │   ├── main.py
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   ├── visualization/
│   │   └── config/
│   ├── requirements.txt
│   └── run.py
├── requirements.txt
├── README.md
├── .gitignore
├── setup.py
└── CONTRIBUTING.md
```

## Configuration

The system can be configured through the `config/config.py` file. Key configuration options include:
- Data preprocessing parameters
- Feature engineering settings
- Model hyperparameters
- Path configurations

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and popular ML libraries
- Uses Streamlit for the interactive dashboard
- Inspired by industry best practices in lead scoring 