import os
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import joblib

from data.preprocess import DataPreprocessor
from features.feature_engineering import FeatureEngineer
from models.train import LeadScoringModel
from models.predict import LeadPredictor
from visualization.visualize import LeadScoringVisualizer
from visualization.dashboard import LeadDashboard
from config.config import get_config

class LeadScoringSystem:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the lead scoring system."""
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.predictor = None
        self.visualizer = LeadScoringVisualizer()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for the system."""
        missing_dirs = []
        for path in self.config['paths'].values():
            if not os.path.exists(path):
                missing_dirs.append(path)
        
        if missing_dirs:
            print("The following directories are required but do not exist:")
            for dir_path in missing_dirs:
                print(f"- {dir_path}")
            
            response = input("\nWould you like to create these directories? (yes/no): ")
            if response.lower() == 'yes':
                for dir_path in missing_dirs:
                    os.makedirs(dir_path, exist_ok=True)
                print("Directories created successfully!")
            else:
                print("Please create the required directories manually before proceeding.")
                print("Required directories:")
                for dir_path in missing_dirs:
                    print(f"- {dir_path}")
                raise FileNotFoundError("Required directories not found. Please create them manually.")
    
    def validate_csv_format(self, df: pd.DataFrame, mode: str = 'train') -> bool:
        """Validate CSV file format based on mode."""
        required_columns = set()
        
        if mode == 'train':
            required_columns = set(self.config['preprocessing']['categorical_columns'] + 
                                 self.config['preprocessing']['numerical_columns'] +
                                 [self.config['preprocessing']['target_column']])
        else:  # predict mode
            required_columns = set(self.config['preprocessing']['categorical_columns'] + 
                                 self.config['preprocessing']['numerical_columns'])
        
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"Error: Missing required columns in CSV file: {missing_columns}")
            print("\nRequired columns:")
            for col in required_columns:
                print(f"- {col}")
            return False
        return True
    
    def prepare_data(self, data_path: str) -> tuple:
        """Prepare data for model training."""
        # Load and validate data
        df = self.preprocessor.load_data(data_path)
        if not self.validate_csv_format(df, mode='train'):
            raise ValueError("Invalid CSV format for training data")
        
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data(
            df,
            self.config['preprocessing']['target_column'],
            self.config['preprocessing']['test_size']
        )
        
        # Engineer features
        X_train = self.feature_engineer.engineer_features(
            pd.DataFrame(X_train, columns=self.preprocessor.feature_names),
            self.config['features']
        )
        X_test = self.feature_engineer.engineer_features(
            pd.DataFrame(X_test, columns=self.preprocessor.feature_names),
            self.config['features']
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   model_type: str = None) -> Dict[str, Any]:
        """Train the lead scoring model."""
        if model_type is None:
            model_type = self.config['model']['default_model']
        
        self.model = LeadScoringModel(model_type=model_type)
        best_params = self.model.train(X_train, y_train, X_val, y_val)
        
        # Save the trained model
        model_path = os.path.join(
            self.config['paths']['models_dir'],
            f'lead_scoring_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        )
        self.model.save_model(model_path)
        
        return best_params
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model."""
        metrics = self.model.evaluate(X_test, y_test)
        
        # Save evaluation results
        results_path = os.path.join(
            self.config['paths']['results_dir'],
            f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def generate_visualizations(self, X_test: np.ndarray, y_test: np.ndarray,
                              feature_importance: pd.DataFrame):
        """Generate and save visualizations."""
        # Plot feature importance
        self.visualizer.plot_feature_importance(feature_importance)
        
        # Plot ROC curve
        y_pred_proba = self.model.model.predict_proba(X_test)[:, 1]
        self.visualizer.plot_roc_curve(y_test, y_pred_proba)
        
        # Plot precision-recall curve
        self.visualizer.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Plot confusion matrix
        y_pred = self.model.model.predict(X_test)
        self.visualizer.plot_confusion_matrix(y_test, y_pred)
    
    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        dashboard = LeadDashboard()
        dashboard.run()
    
    def predict_from_csv(self, csv_path: str, model_path: str) -> List[Dict[str, Any]]:
        """Generate predictions from a CSV file."""
        # Load and validate data
        df = pd.read_csv(csv_path)
        if not self.validate_csv_format(df, mode='predict'):
            raise ValueError("Invalid CSV format for prediction data")
        
        # Convert DataFrame to list of dictionaries
        leads_data = df.to_dict('records')
        
        # Generate predictions
        predictions = self.predict_leads(leads_data, model_path)
        
        # Save predictions
        predictions_path = os.path.join(
            self.config['paths']['results_dir'],
            f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        self.predictor.save_predictions(predictions, predictions_path)
        
        return predictions
    
    def predict_leads(self, leads_data: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
        """Generate predictions for new leads."""
        self.predictor = LeadPredictor(model_path)
        predictions = self.predictor.batch_predict(leads_data)
        
        # Save predictions
        predictions_path = os.path.join(
            self.config['paths']['results_dir'],
            f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        self.predictor.save_predictions(predictions, predictions_path)
        
        return predictions

def train_model(data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the lead scoring model."""
    # Initialize components
    print("\n1. INITIALIZING COMPONENTS")
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    model = LeadScoringModel(model_type='xgboost')
    
    # Prepare data - now returns DataFrames instead of numpy arrays
    print("\n2. PREPARING DATA")
    X_train_df, X_test_df, y_train, y_test = preprocessor.prepare_data(data_path)
    
    print(f"X_train_df shape: {X_train_df.shape}, columns: {X_train_df.columns.tolist()}")
    print(f"X_test_df shape: {X_test_df.shape}, columns: {X_test_df.columns.tolist()}")
    print(f"y_train shape: {y_train.shape}, class counts: {np.bincount(y_train)}")
    print(f"y_test shape: {y_test.shape}, class counts: {np.bincount(y_test)}")
    
    # Engineer features
    print("\n3. ENGINEERING FEATURES")
    X_train_df = feature_engineer.engineer_features(X_train_df)
    X_test_df = feature_engineer.engineer_features(X_test_df)
    
    print(f"After feature engineering - X_train_df shape: {X_train_df.shape}")
    print(f"After feature engineering - X_test_df shape: {X_test_df.shape}")
    
    # Verify we have features
    if X_train_df.shape[1] == 0:
        raise ValueError("After feature engineering, X_train_df has 0 features!")
    
    # Convert to numpy arrays for model training
    X_train = X_train_df.values
    X_test = X_test_df.values
    
    # Store feature names in the model
    model.feature_names = X_train_df.columns.tolist()
    print(f"Feature names: {model.feature_names}")
    
    # Train model (only pass X_train and y_train)
    print("\n4. TRAINING MODEL")
    best_params = model.train(X_train, y_train)
    
    # Evaluate model
    print("\n5. EVALUATING MODEL")
    metrics = model.evaluate(X_test, y_test)
    
    # Save model with feature names
    print("\n6. SAVING MODEL")
    model_data = {
        'model': model.model,
        'feature_names': X_train_df.columns.tolist(),
        'best_params': best_params
    }
    
    # Save model
    model_dir = config.get('paths', {}).get('model_dir', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 
                            f'lead_scoring_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
    try:
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    return metrics

def predict_from_csv(data_path: str, model_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate predictions from CSV data."""
    # Initialize components
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    predictor = LeadPredictor(model_path)
    
    # Prepare data
    X, _, _, _ = preprocessor.prepare_data(data_path)
    
    # Engineer features
    X = feature_engineer.engineer_features(pd.DataFrame(X))
    
    # Generate predictions
    predictions = predictor.batch_predict(X.to_dict('records'))
    
    # Save predictions
    output_path = os.path.join(config['paths']['output_dir'], 
                             f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    predictor.save_predictions(predictions, output_path)
    
    return predictions

def run_dashboard():
    """Run the Streamlit dashboard."""
    dashboard = LeadDashboard()
    dashboard.run()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lead Scoring System')
    parser.add_argument('--mode', choices=['train', 'predict', 'dashboard'],
                      required=True, help='Operation mode')
    parser.add_argument('--data', help='Path to input CSV file')
    parser.add_argument('--model', help='Path to trained model file')
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    try:
        if args.mode == 'train':
            if not args.data:
                raise ValueError("Data file path is required for training mode")
            print(f"Training model using data from: {args.data}")
            metrics = train_model(args.data, config)
            print("Model training completed successfully!")
            print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
        elif args.mode == 'predict':
            if not args.model or not args.data:
                raise ValueError("Both model and data file paths are required for prediction mode")
            print(f"Generating predictions for: {args.data}")
            predictions = predict_from_csv(args.data, args.model, config)
            print(f"Generated {len(predictions)} predictions")
            
        elif args.mode == 'dashboard':
            print("Starting dashboard...")
            run_dashboard()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main() 