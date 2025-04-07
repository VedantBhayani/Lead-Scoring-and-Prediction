import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import optuna
from typing import Dict, Any, Tuple, List
import joblib
import os

class LeadScoringModel:
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.best_params = None
        self.feature_names = None
        
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate that the data has enough samples and classes."""
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Error: Only one class ({unique_classes[0]}) present in the data. Need at least two classes for binary classification.")
            return False
        if len(y) < 10:
            print("Error: Not enough samples for training. Need at least 10 samples.")
            return False
        print(f"\nClass distribution:")
        for cls in unique_classes:
            print(f"Class {cls}: {np.sum(y == cls)} samples")
        return True
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model with hyperparameter optimization using cross-validation."""
        print("Training Random Forest model...")
        def objective(trial):
            try:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3)
                }
                
                # Use 5-fold cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train = X_train[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val = X_train[val_idx]
                    y_fold_val = y_train[val_idx]
                    
                    model = RandomForestClassifier(**params, random_state=42)
                    model.fit(X_fold_train, y_fold_train)
                    
                    y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                    scores.append(score)
                
                return np.mean(scores)
            except Exception as e:
                print(f"Trial failed: {str(e)}")
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        if not study.trials_dataframe().empty:
            self.best_params = study.best_params
            self.model = RandomForestClassifier(**self.best_params, random_state=42)
            self.model.fit(X_train, y_train)
            return study.best_params
        else:
            raise ValueError("No successful trials completed")
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model with hyperparameter optimization using cross-validation."""
        print("Training XGBoost model...")
        def objective(trial):
            try:
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.8)
                }
                
                # Use 5-fold cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train = X_train[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val = X_train[val_idx]
                    y_fold_val = y_train[val_idx]
                    
                    model = xgb.XGBClassifier(**params, random_state=42)
                    model.fit(X_fold_train, y_fold_train)
                    
                    y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                    scores.append(score)
                
                return np.mean(scores)
            except Exception as e:
                print(f"Trial failed: {str(e)}")
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        if not study.trials_dataframe().empty:
            self.best_params = study.best_params
            self.model = xgb.XGBClassifier(**self.best_params, random_state=42)
            self.model.fit(X_train, y_train)
            return study.best_params
        else:
            raise ValueError("No successful trials completed")
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM model with hyperparameter optimization using cross-validation."""
        print("Training LightGBM model...")
        def objective(trial):
            try:
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.8)
                }
                
                # Use 5-fold cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_fold_train = X_train[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val = X_train[val_idx]
                    y_fold_val = y_train[val_idx]
                    
                    model = lgb.LGBMClassifier(**params, random_state=42)
                    model.fit(X_fold_train, y_fold_train)
                    
                    y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                    scores.append(score)
                
                return np.mean(scores)
            except Exception as e:
                print(f"Trial failed: {str(e)}")
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        if not study.trials_dataframe().empty:
            self.best_params = study.best_params
            self.model = lgb.LGBMClassifier(**self.best_params, random_state=42)
            self.model.fit(X_train, y_train)
            return study.best_params
        else:
            raise ValueError("No successful trials completed")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the selected model type."""
        print(f"\nStarting model training with {self.model_type}...")
        
        # Validate data
        if not self.validate_data(X_train, y_train):
            raise ValueError("Training data validation failed")
        
        if self.model_type == 'random_forest':
            return self.train_random_forest(X_train, y_train)
        elif self.model_type == 'xgboost':
            return self.train_xgboost(X_train, y_train)
        elif self.model_type == 'lightgbm':
            return self.train_lightgbm(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        print("\nEvaluating model performance...")
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate various metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision_recall_curve': precision_recall_curve(y_test, y_pred_proba)
        }
        
        print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        elif self.model_type in ['xgboost', 'lightgbm']:
            importance = self.model.feature_importances_
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def save_model(self, model_path: str):
        """Save the trained model."""
        print(f"\nSaving model to {model_path}...")
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        joblib.dump(model_data, model_path)
        print("Model saved successfully!")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        print(f"\nLoading model from {model_path}...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        print("Model loaded successfully!")

def main():
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Load processed data
        print("Loading processed data...")
        X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(processed_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(processed_dir, 'y_test.npy'))
        
        # Print data shapes
        print(f"\nData shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Define feature names
        feature_names = ['company_size', 'lead_source', 'industry', 'country',
                        'website_visits', 'email_opens', 'form_submissions', 'time_on_site']
        
        # Train model
        model = LeadScoringModel(model_type='xgboost')
        model.feature_names = feature_names
        best_params = model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        # Save model
        model_path = os.path.join(models_dir, 'lead_scoring_model.joblib')
        model.save_model(model_path)
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")

if __name__ == "__main__":
    main() 