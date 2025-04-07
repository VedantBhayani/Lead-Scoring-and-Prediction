import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from datetime import datetime

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        self.categorical_columns = self.config.get('categorical_columns', 
                                            ['company_size', 'lead_source', 'industry', 'country'])
        self.numerical_columns = self.config.get('numerical_columns', 
                                          ['website_visits', 'email_opens', 'form_submissions', 'time_on_site'])
        self.feature_names = []
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features including a simple engagement score."""
        # Create a copy to avoid modifying the original
        df_new = df.copy()
        
        # Check if the required numerical columns exist
        num_cols = [col for col in self.numerical_columns if col in df.columns]
        
        # Only create engagement score if we have at least one numerical column
        if num_cols:
            # Create a simple engagement score as the sum of all numerical features
            df_new['engagement_score'] = df[num_cols].sum(axis=1)
            
        # Check if categorical columns exist and create count features
        cat_cols = [col for col in self.categorical_columns if col in df.columns]
        if cat_cols:
            # Create a categorical presence indicator
            df_new['has_categorical_data'] = (df[cat_cols].notna().sum(axis=1) > 0).astype(int)
            
        return df_new
        
    def engineer_features(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply all feature engineering steps to either DataFrame or numpy array."""
        # Convert numpy array to DataFrame if needed
        if isinstance(X, np.ndarray):
            # Create default column names if X is a numpy array
            columns = []
            if self.numerical_columns:
                columns.extend(self.numerical_columns[:X.shape[1]])
            if len(columns) < X.shape[1]:
                # Add generic column names for any remaining columns
                for i in range(len(columns), X.shape[1]):
                    columns.append(f'feature_{i}')
                    
            df = pd.DataFrame(X, columns=columns[:X.shape[1]])
        else:
            df = X.copy()
            
        print(f"Feature engineering input shape: {df.shape}")
        
        # For very small datasets, just maintain the existing features
        if df.shape[0] < 10:
            print("Small dataset detected, minimal feature engineering applied")
            df_engineered = self.create_basic_features(df)
        else:
            # Apply more complex feature engineering for larger datasets
            df_engineered = self.create_basic_features(df)
        
        print(f"Feature engineering output shape: {df_engineered.shape}")
        
        # Ensure no NaN values in the engineered features
        df_engineered = df_engineered.fillna(0)
        
        # Store feature names
        self.feature_names = df_engineered.columns.tolist()
        
        return df_engineered
    
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