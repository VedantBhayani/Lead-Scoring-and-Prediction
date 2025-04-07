import numpy as np
import pandas as pd
from typing import Dict, Any, List
import joblib
from datetime import datetime

class LeadPredictor:
    def __init__(self, model_path: str):
        """Initialize the predictor with a trained model."""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        self.best_params = self.model_data['best_params']
    
    def preprocess_input(self, lead_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data to match model requirements."""
        # Convert input data to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        return df.values
    
    def predict_proba(self, lead_data: Dict[str, Any]) -> float:
        """Predict conversion probability for a lead."""
        X = self.preprocess_input(lead_data)
        return self.model.predict_proba(X)[0][1]
    
    def predict(self, lead_data: Dict[str, Any], threshold: float = 0.5) -> bool:
        """Predict whether a lead will convert."""
        proba = self.predict_proba(lead_data)
        return proba >= threshold
    
    def get_lead_score(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive lead score with additional information."""
        proba = self.predict_proba(lead_data)
        prediction = self.predict(lead_data)
        
        # Calculate lead score (0-100)
        lead_score = int(proba * 100)
        
        # Determine lead category
        if lead_score >= 80:
            category = "Hot"
        elif lead_score >= 50:
            category = "Warm"
        else:
            category = "Cold"
        
        # Get feature importance for this lead
        feature_importance = self.get_feature_importance(lead_data)
        
        return {
            'lead_score': lead_score,
            'conversion_probability': proba,
            'predicted_conversion': prediction,
            'category': category,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_feature_importance(self, lead_data: Dict[str, Any]) -> Dict[str, float]:
        """Get feature importance scores for the current lead."""
        X = self.preprocess_input(lead_data)
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            # For models without direct feature importance
            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def batch_predict(self, leads_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate predictions for multiple leads."""
        return [self.get_lead_score(lead) for lead in leads_data]
    
    def save_predictions(self, predictions: List[Dict[str, Any]], output_path: str):
        """Save predictions to a CSV file."""
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

def main():
    # Example usage
    model_path = "models/lead_scoring_model.joblib"
    
    # Sample lead data
    lead_data = {
        'company_size': '51-200',
        'website_visits': 5,
        'email_opens': 3,
        'form_submissions': 1,
        'created_at_hour': 14,
        'created_at_day_of_week': 2,
        'created_at_month': 3,
        'created_at_quarter': 1,
        'company_size_score': 3,
        'lead_source_score': 2,
        'engagement_score': 0.75
    }
    
    # Initialize predictor
    predictor = LeadPredictor(model_path)
    
    # Generate prediction
    result = predictor.get_lead_score(lead_data)
    
    print("Lead Scoring Results:")
    print(f"Lead Score: {result['lead_score']}")
    print(f"Category: {result['category']}")
    print(f"Conversion Probability: {result['conversion_probability']:.2f}")
    print(f"Predicted Conversion: {result['predicted_conversion']}")
    print("\nTop 3 Important Features:")
    for feature, importance in list(result['feature_importance'].items())[:3]:
        print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main() 