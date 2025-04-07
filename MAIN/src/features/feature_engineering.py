import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categorical_columns = config['categorical_columns']
        self.numerical_columns = config['numerical_columns']
        self.feature_names = []
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['created_at_hour'] = df['created_at'].dt.hour
            df['created_at_day_of_week'] = df['created_at'].dt.dayofweek
            df['created_at_month'] = df['created_at'].dt.month
            df['created_at_quarter'] = df['created_at'].dt.quarter
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        for i, col1 in enumerate(self.numerical_columns):
            for col2 in self.numerical_columns[i+1:]:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features."""
        # Engagement score
        df['engagement_score'] = (
            df['website_visits'] * 0.4 +
            df['email_opens'] * 0.3 +
            df['form_submissions'] * 0.3
        )
        
        # Company size score
        size_mapping = {
            '1-10': 1,
            '11-50': 2,
            '51-200': 3,
            '201-500': 4,
            '501-1000': 5,
            '1001-5000': 6,
            '5001-10000': 7,
            '10001+': 8
        }
        df['company_size_score'] = df['company_size'].map(size_mapping)
        
        # Lead source score
        source_mapping = {
            'Website': 1,
            'Referral': 2,
            'Social Media': 3,
            'Email Campaign': 4,
            'Trade Show': 5
        }
        df['lead_source_score'] = df['lead_source'].map(source_mapping)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = df.copy()
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create aggregation features
        df = self.create_aggregation_features(df)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return the list of engineered feature names."""
        return self.feature_names

def main():
    # Example usage
    config = {
        'categorical_columns': ['company_size', 'lead_source'],
        'numerical_columns': ['website_visits', 'email_opens', 'form_submissions']
    }
    engineer = FeatureEngineer(config)
    
    # Load sample data
    df = pd.read_csv('data/leads.csv')
    
    # Engineer features
    df_engineered = engineer.engineer_features(df)
    
    print("Feature engineering completed successfully!")
    print(f"Number of features: {len(engineer.get_feature_names())}")
    print("Feature names:", engineer.get_feature_names())

if __name__ == "__main__":
    main() 