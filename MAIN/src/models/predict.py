import pandas as pd
import numpy as np
from typing import Dict, Any, List
import joblib
from datetime import datetime
import json
import os

class LeadPredictor:
    def __init__(self, model_dir: str):
        """Initialize the predictor with the most recent trained model."""
        # Find the most recent model file
        model_files = [f for f in os.listdir(model_dir) if f.startswith('lead_scoring_model_') and f.endswith('.joblib')]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        latest_model = max(model_files)
        model_path = os.path.join(model_dir, latest_model)
        print(f"Loading model from: {model_path}")
        
        # Load the model
        model_data = joblib.load(model_path)
        
        # Extract model components
        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.preprocessor = model_data.get('preprocessor')
            self.feature_names = model_data.get('feature_names')
            print("Model components loaded successfully")
        else:
            self.model = model_data
            self.preprocessor = None
            self.feature_names = None
            print("Warning: Model loaded in legacy format")
        
        if not self.model:
            raise ValueError("Model not found in the loaded file")
    
    def preprocess_data(self, leads_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        # Make a copy to avoid modifying original data
        df = leads_df.copy()
        
        # Map email_opens to email_interactions if present
        if 'email_opens' in df.columns and 'email_interactions' not in df.columns:
            df['email_interactions'] = df['email_opens']
            
        # Calculate pages_viewed based on time_on_site (assuming average 2 minutes per page)
        if 'time_on_site' in df.columns and 'pages_viewed' not in df.columns:
            df['pages_viewed'] = (df['time_on_site'] / 2).round()
            
        # Estimate social_media_engagement if missing (based on website_visits)
        if 'website_visits' in df.columns and 'social_media_engagement' not in df.columns:
            df['social_media_engagement'] = (df['website_visits'] * 0.3).round()  # Assume 30% of visits come from social
        
        # Define expected features and their types
        expected_features = {
            'company_size': 'category',
            'lead_source': 'category',
            'industry': 'category',
            'country': 'category',
            'website_visits': 'float',
            'form_submissions': 'float',
            'email_interactions': 'float',
            'time_on_site': 'float',
            'pages_viewed': 'float',
            'social_media_engagement': 'float'
        }
        
        # Ensure all expected features exist
        for feature, dtype in expected_features.items():
            if feature not in df.columns:
                print(f"Warning: Missing feature {feature}, filling with 0")
                if dtype == 'category':
                    df[feature] = 'unknown'
                else:
                    df[feature] = 0
        
        # Convert types
        for feature, dtype in expected_features.items():
            if dtype == 'category':
                df[feature] = df[feature].astype('category')
            else:
                df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
        
        # Keep only required features in the correct order
        df = df[list(expected_features.keys())]
        
        # If preprocessor is available, use it
        if self.preprocessor:
            try:
                processed_df = self.preprocessor.transform_new_data(df)
                return processed_df
            except Exception as e:
                print(f"Warning: Error in preprocessor: {str(e)}")
                # If preprocessor fails, do basic preprocessing
                return df
        
        return df
    
    def predict_single(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for a single lead."""
        # Convert to DataFrame
        lead_df = pd.DataFrame([lead_data])
        
        # Preprocess
        X = self.preprocess_data(lead_df)
        
        # Generate prediction
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X, enable_categorical=True)[0]
                lead_score = proba[1] * 100
            else:
                pred = self.model.predict(X, enable_categorical=True)[0]
                lead_score = float(pred) * 100
        except Exception as e:
            print(f"Warning: Error in prediction: {str(e)}")
            # Try without enable_categorical
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                lead_score = proba[1] * 100
            else:
                pred = self.model.predict(X)[0]
                lead_score = float(pred) * 100
        
        conversion_probability = lead_score / 100
        
        # Determine category
        if lead_score >= 80:
            category = "Hot"
        elif lead_score >= 50:
            category = "Warm"
        else:
            category = "Cold"
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            features = self.feature_names or X.columns
            importance = self.model.feature_importances_
            feature_importance = dict(zip(features, importance))
        
        return {
            'lead_score': lead_score,
            'conversion_probability': conversion_probability,
            'predicted_conversion': conversion_probability >= 0.5,
            'category': category,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_predict(self, leads_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for multiple leads and preserve original data."""
        # Keep a copy of original data
        original_data = leads_df.copy()
        
        # Preprocess features for prediction
        X = self.preprocess_data(leads_df)
        
        # Generate predictions
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X, enable_categorical=True)[:, 1]
            else:
                predictions = self.model.predict(X, enable_categorical=True)
                probabilities = predictions.astype(float)
        except Exception as e:
            print(f"Warning: Error with enable_categorical: {str(e)}")
            # Try without enable_categorical
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[:, 1]
            else:
                predictions = self.model.predict(X)
                probabilities = predictions.astype(float)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'lead_score': probabilities * 100,
            'conversion_probability': probabilities,
            'predicted_conversion': probabilities >= 0.5
        })
        
        # Add category based on lead score
        def get_category(score):
            if score >= 80:
                return "Hot"
            elif score >= 50:
                return "Warm"
            return "Cold"
        
        predictions_df['category'] = predictions_df['lead_score'].apply(get_category)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            features = self.feature_names or X.columns
            importance = self.model.feature_importances_
            feature_importance = dict(zip(features, importance))
            predictions_df['feature_importance'] = [feature_importance] * len(predictions_df)
        
        # Add timestamp
        predictions_df['timestamp'] = datetime.now().isoformat()
        
        # Combine original data with predictions
        # Reset index to ensure proper alignment
        original_data.reset_index(drop=True, inplace=True)
        predictions_df.reset_index(drop=True, inplace=True)
        
        # Combine predictions with original data
        result_df = pd.concat([original_data, predictions_df], axis=1)
        
        return result_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_path: str):
        """Save predictions to a CSV file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save predictions
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

def main():
    # Set up paths relative to the script location
    base_dir = os.getcwd()  # Use current directory as base
    
    model_dir = os.path.join(base_dir, "MAIN", "models")
    output_dir = os.path.join(base_dir, "MAIN", "output")
    input_csv = os.path.join(base_dir, "MAIN", "data", "leads_for_prediction.csv")
    
    # Validate paths
    if not os.path.exists(model_dir):
        print(f"Error: Models directory not found at {model_dir}")
        return
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV not found at {input_csv}")
        return
    
    try:
        # Initialize predictor
        predictor = LeadPredictor(model_dir)
        
        # Load leads data
        leads_df = pd.read_csv(input_csv)
        print(f"Loaded {len(leads_df)} leads from {input_csv}")
        
        # Generate predictions
        predictions_df = predictor.batch_predict(leads_df)
        print(f"Generated predictions for {len(predictions_df)} leads")
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"predictions_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save predictions
        predictor.save_predictions(predictions_df, output_path)
        
        print("\nPrediction Results (first 5 rows):")
        print(predictions_df[['lead_name', 'lead_score', 'category', 'conversion_probability']].head().to_string())
        
    except Exception as e:
        print(f"Error during prediction process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 