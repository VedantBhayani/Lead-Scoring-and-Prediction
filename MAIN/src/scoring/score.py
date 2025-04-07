import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, List
import json

class LeadScorer:
    def __init__(self, model_path: str):
        """Initialize the lead scorer with a trained model."""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        self.best_params = self.model_data['best_params']
        
    def preprocess_lead(self, lead_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single lead's data for prediction."""
        # Create a DataFrame with the lead data
        df = pd.DataFrame([lead_data])
        
        # Handle categorical variables
        categorical_features = ['company_size', 'lead_source', 'industry', 'country']
        for feature in categorical_features:
            if feature in df.columns:
                # Use the same encoding as in training
                df[feature] = df[feature].astype('category').cat.codes
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training data
        X = df[self.feature_names].values
        return X
    
    def predict_proba(self, lead_data: Dict[str, Any]) -> float:
        """Predict the probability of conversion for a lead."""
        X = self.preprocess_lead(lead_data)
        proba = self.model.predict_proba(X)[0][1]
        return float(proba)
    
    def predict(self, lead_data: Dict[str, Any]) -> int:
        """Predict whether a lead will convert (1) or not (0)."""
        X = self.preprocess_lead(lead_data)
        prediction = self.model.predict(X)[0]
        return int(prediction)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'models', 'lead_scoring_model.joblib')
    data_path = os.path.join(base_dir, 'data', 'sample_leads.csv')
    
    try:
        # Initialize scorer
        print("Loading trained model...")
        scorer = LeadScorer(model_path)
        print("Model loaded successfully!")
        
        # Load all leads
        print("\nLoading leads from sample data...")
        leads_df = pd.read_csv(data_path)
        print(f"Loaded {len(leads_df)} leads")
        
        # Process each lead
        print("\nProcessing leads and making predictions...")
        print("-" * 80)
        
        results = []
        for idx, lead in leads_df.iterrows():
            lead_data = lead.to_dict()
            proba = scorer.predict_proba(lead_data)
            prediction = scorer.predict(lead_data)
            
            result = {
                'Lead ID': idx + 1,
                'Company Size': lead_data['company_size'],
                'Industry': lead_data['industry'],
                'Lead Source': lead_data['lead_source'],
                'Website Visits': lead_data['website_visits'],
                'Email Opens': lead_data['email_opens'],
                'Time on Site': lead_data['time_on_site'],
                'Conversion Probability': f"{proba:.2%}",
                'Predicted Outcome': 'Converted' if prediction == 1 else 'Not Converted',
                'Actual Outcome': 'Converted' if lead_data['converted'] == 1 else 'Not Converted'
            }
            results.append(result)
            
            # Print individual lead result
            print(f"\nLead {idx + 1}:")
            print(f"Company: {lead_data['company_size']} | Industry: {lead_data['industry']}")
            print(f"Engagement: {lead_data['website_visits']} visits, {lead_data['email_opens']} email opens, {lead_data['time_on_site']}s on site")
            print(f"Prediction: {proba:.2%} chance of conversion")
            print(f"Predicted: {'Converted' if prediction == 1 else 'Not Converted'}")
            print(f"Actual: {'Converted' if lead_data['converted'] == 1 else 'Not Converted'}")
            print("-" * 40)
        
        # Convert results to DataFrame for summary
        results_df = pd.DataFrame(results)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 80)
        print(f"Total Leads: {len(results_df)}")
        print(f"Average Conversion Probability: {results_df['Conversion Probability'].str.rstrip('%').astype('float').mean():.2f}%")
        print(f"Predicted Conversions: {len(results_df[results_df['Predicted Outcome'] == 'Converted'])}")
        print(f"Actual Conversions: {len(results_df[results_df['Actual Outcome'] == 'Converted'])}")
        
        # Calculate accuracy
        correct_predictions = len(results_df[results_df['Predicted Outcome'] == results_df['Actual Outcome']])
        accuracy = correct_predictions / len(results_df)
        print(f"Prediction Accuracy: {accuracy:.2%}")
        
        # Save results to CSV
        output_path = os.path.join(base_dir, 'data', 'lead_predictions.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during scoring: {str(e)}")

if __name__ == "__main__":
    main() 