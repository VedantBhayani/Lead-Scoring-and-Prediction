import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import os

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categorical_columns = config['categorical_columns']
        self.numerical_columns = config['numerical_columns']
        self.target_column = config['target_column']
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            print(f"Data shape: {df.shape}")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate data format and required columns."""
        required_columns = self.categorical_columns + self.numerical_columns
        if self.target_column in df.columns:
            required_columns.append(self.target_column)
            
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        print("Data validation successful")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        print("Starting data preprocessing...")
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed[self.numerical_columns] = df_processed[self.numerical_columns].fillna(0)
        df_processed[self.categorical_columns] = df_processed[self.categorical_columns].fillna('Unknown')
        print("Handled missing values")
        
        # Convert categorical variables to numeric
        for col in self.categorical_columns:
            df_processed[col] = pd.Categorical(df_processed[col]).codes
        print("Encoded categorical variables")
            
        return df_processed
    
    def prepare_data(self, file_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load, validate, preprocess and split data."""
        # Load data
        df = self.load_data(file_path)
        
        # Validate data
        self.validate_data(df)
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Prepare features and target
        X = df_processed[self.categorical_columns + self.numerical_columns].values
        y = df_processed[self.target_column].values if self.target_column in df.columns else None
        
        # Split data if target is available
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print(f"Data split into training ({X_train.shape}) and test ({X_test.shape}) sets")
            return X_train, X_test, y_train, y_test
        else:
            return X, None, None, None
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using Label Encoding."""
        df_encoded = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df_encoded[column] = self.label_encoders[column].fit_transform(df_encoded[column])
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_scaled = df.copy()
        df_scaled[numeric_columns] = self.scaler.fit_transform(df_scaled[numeric_columns])
        return df_scaled
    
    def save_preprocessor(self, file_path: str):
        """Save preprocessor state for later use."""
        preprocessor_state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        # Save the state (implementation depends on your storage method)
        pass
    
    def load_preprocessor(self, file_path: str):
        """Load preprocessor state."""
        # Load the state (implementation depends on your storage method)
        pass

def main():
    # Configuration
    config = {
        'categorical_columns': ['company_size', 'lead_source', 'industry', 'country'],
        'numerical_columns': ['website_visits', 'email_opens', 'form_submissions', 'time_on_site'],
        'target_column': 'converted'
    }
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    input_file = os.path.join(data_dir, 'sample_leads.csv')
    output_dir = os.path.join(data_dir, 'processed')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process data
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(input_file)
        
        # Save processed data
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        print("\nData preprocessing completed successfully!")
        print(f"Processed data saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main() 