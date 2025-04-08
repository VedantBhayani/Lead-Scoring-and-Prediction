# Lead Scoring Prediction System

A comprehensive machine learning system for predicting lead scores using various features and advanced modeling techniques.

## Overview

This system provides an end-to-end solution for lead scoring, including:
- Data preprocessing and feature engineering
- Model training with XGBoost
- Model evaluation and visualization
- Batch prediction capabilities

## Features

- **Data Preprocessing**: Automated data cleaning and transformation
- **Feature Engineering**: Advanced feature creation and selection
- **Model Training**: XGBoost model with hyperparameter optimization
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Batch Prediction**: Support for processing multiple leads at once

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lead-scoring-prediction.git
cd lead-scoring-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv lead_scoring
lead_scoring\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
lead-scoring-prediction/
├── MAIN/
│   ├── src/
│   │   ├── models/
│   │   │   ├── train.py
│   │   │   └── predict.py
│   │   └── data/
│   │       └── leads_for_prediction.csv
│   ├── models/
│   │   └── lead_scoring_model_*.joblib
│   ├── data/
│   │   └── leads_for_prediction.csv
│   └── output/
│       └── predictions_*.csv
├── requirements.txt
└── README.md
```

## Usage

### Training the Model

1. Prepare your training data in CSV format with the following columns:
   - company_size
   - lead_source
   - industry
   - country
   - website_visits
   - form_submissions
   - email_interactions
   - time_on_site
   - pages_viewed
   - social_media_engagement

2. Run the training script:
```bash
python MAIN/src/models/train.py
```

### Making Predictions

1. Prepare your leads data in CSV format with at least these columns:
   - lead_id
   - lead_name
   - company_size
   - lead_source
   - industry
   - country
   - website_visits
   - email_opens (or email_interactions)
   - form_submissions
   - time_on_site

2. Place your leads data in `MAIN/data/leads_for_prediction.csv`

3. Run the prediction script:
```bash
python MAIN/src/models/predict.py
```

4. Find the predictions in the `MAIN/output/` directory with timestamp in the filename

## Output Format

The prediction output includes:
- lead_name: Name of the lead
- lead_score: Score from 0-100
- category: "Hot" (≥80), "Warm" (≥50), or "Cold" (<50)
- conversion_probability: Probability of conversion

## Troubleshooting

Common issues and solutions:

1. **Missing Features**: The system will automatically handle missing features by:
   - Converting email_opens to email_interactions
   - Calculating pages_viewed from time_on_site
   - Estimating social_media_engagement from website_visits

2. **Data Type Errors**: Ensure categorical columns (company_size, lead_source, industry, country) are properly formatted

3. **Model Loading**: Make sure the model file exists in the MAIN/models directory

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and popular ML libraries
- Inspired by industry best practices in lead scoring 