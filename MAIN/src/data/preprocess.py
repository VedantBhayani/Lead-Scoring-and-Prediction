import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, Union
import os
import json

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Default column configurations that match our sample data
        self.categorical_columns = self.config.get('categorical_columns', 
                                                 ['company_size', 'lead_source', 'industry', 'country'])
        self.numerical_columns = self.config.get('numerical_columns', 
                                               ['website_visits', 'email_opens', 'form_submissions', 'time_on_site'])
        self.target_column = self.config.get('target_column', 'converted')
        self.name_column = self.config.get('name_column', 'lead_name')  # Add name column configuration
        self.id_column = self.config.get('id_column', 'lead_id')  # Add ID column configuration
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []  # Store feature names
        self.name_mapping = {}  # Store mapping between lead IDs and names
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded data from {file_path}")
            print(f"Data shape: {df.shape}")
            
            # Create ID column if it doesn't exist
            if self.id_column not in df.columns:
                df[self.id_column] = range(len(df))
                print(f"Created {self.id_column} column with sequential IDs")
            
            # Create name column if it doesn't exist
            if self.name_column not in df.columns:
                df[self.name_column] = [f"Lead_{i}" for i in range(len(df))]
                print(f"Created {self.name_column} column with default names")
            
            # Create name mapping
            self.name_mapping = dict(zip(df[self.id_column], df[self.name_column]))
            print(f"Created name mapping for {len(self.name_mapping)} leads")
            
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
        
        # Store name mapping before dropping name columns
        if self.id_column in df_processed.columns and self.name_column in df_processed.columns:
            self.name_mapping = dict(zip(df_processed[self.id_column], df_processed[self.name_column]))
        
        # Drop name and ID columns as they shouldn't be used as features
        columns_to_drop = [col for col in [self.name_column, self.id_column] if col in df_processed.columns]
        if columns_to_drop:
            df_processed = df_processed.drop(columns=columns_to_drop)
            print(f"Dropped columns: {columns_to_drop}")
        
        # Handle missing values
        for col in self.numerical_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0)
        
        for col in self.categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        print("Handled missing values")
        
        # Convert categorical variables to numeric while preserving column names
        for col in self.categorical_columns:
            if col in df_processed.columns:
                # Create a label encoder for this column if it doesn't exist
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                # Fit and transform the column
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        print("Encoded categorical variables")
        
        # Scale numerical features
        numerical_present = [col for col in self.numerical_columns if col in df_processed.columns]
        if numerical_present:
            df_processed[numerical_present] = self.scaler.fit_transform(df_processed[numerical_present])
            print("Scaled numerical features")
        
        # Store feature names (all columns except target)
        self.feature_names = [col for col in df_processed.columns if col != self.target_column]
            
        return df_processed
    
    def prepare_data(self, file_path_or_df: Union[str, pd.DataFrame], test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load, validate, preprocess and split data.
        
        Returns:
            X_train_df, X_test_df, y_train, y_test: DataFrames for X and numpy arrays for y
        """
        # Load data
        if isinstance(file_path_or_df, str):
            df = self.load_data(file_path_or_df)
        else:
            df = file_path_or_df.copy()
        
        # Validate data
        self.validate_data(df)
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Get available feature columns (must be a non-empty list)
        feature_columns = self.categorical_columns + self.numerical_columns
        available_columns = [col for col in feature_columns if col in df_processed.columns]
        
        if not available_columns:
            raise ValueError(f"No valid feature columns found in the data. Available columns: {df_processed.columns.tolist()}")
            
        # Prepare features and target
        X_df = df_processed[available_columns]
        y = df_processed[self.target_column].values if self.target_column in df_processed.columns else None
        
        # Store the feature names for later use
        self.feature_names = available_columns
        
        # Check X_df has features
        if X_df.shape[1] == 0:
            raise ValueError(f"Processed data has 0 features. Available columns: {df_processed.columns.tolist()}")
        
        # Split data if target is available
        if y is not None:
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_df, y, test_size=test_size, random_state=42
            )
            print(f"Data split into training {X_train_df.shape} and test {X_test_df.shape} sets")
            return X_train_df, X_test_df, y_train, y_test
        else:
            return X_df, None, None, None
    
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
    
    def get_feature_names(self) -> List[str]:
        """Return the list of feature names."""
        return self.feature_names
    
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
    
    def get_name_mapping(self) -> Dict[Any, str]:
        """Return the mapping between lead IDs and names."""
        return self.name_mapping
    
    def save_name_mapping(self, output_dir: str):
        """Save name mapping to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        mapping_file = os.path.join(output_dir, 'name_mapping.json')
        # Convert all keys to strings for JSON compatibility
        string_mapping = {str(k): v for k, v in self.name_mapping.items()}
        
        with open(mapping_file, 'w') as f:
            json.dump(string_mapping, f, indent=4)
        print(f"Name mapping saved to: {mapping_file}")
    
    def load_name_mapping(self, file_path: str):
        """Load name mapping from a JSON file."""
        with open(file_path, 'r') as f:
            string_mapping = json.load(f)
        # Convert string keys back to original type if needed
        self.name_mapping = {int(k): v for k, v in string_mapping.items()}
        print(f"Loaded name mapping for {len(self.name_mapping)} leads")

def main():
    # Configuration
    config = {
        'categorical_columns': ['company_size', 'lead_source', 'industry', 'country'],
        'numerical_columns': ['website_visits', 'email_opens', 'form_submissions', 'time_on_site'],
        'target_column': 'converted',
        'name_column': 'lead_name',
        'id_column': 'lead_id'
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
        X_train_df, X_test_df, y_train, y_test = preprocessor.prepare_data(input_file)
        
        # Save processed data
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train_df.values)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test_df.values)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save feature names
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(preprocessor.get_feature_names()))
        
        # Save name mapping
        preprocessor.save_name_mapping(output_dir)
        
        print("\nData preprocessing completed successfully!")
        print(f"Processed data saved to: {output_dir}")
        print(f"Number of features: {len(preprocessor.get_feature_names())}")
        print(f"Number of leads with names: {len(preprocessor.get_name_mapping())}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main() 