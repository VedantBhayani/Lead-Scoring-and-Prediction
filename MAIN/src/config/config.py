"""Configuration settings for the Lead Scoring System."""

import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get configuration settings."""
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    config = {
        'paths': {
            'data_dir': os.path.join(base_dir, 'data'),
            'models_dir': os.path.join(base_dir, 'models'),
            'results_dir': os.path.join(base_dir, 'results'),
            'output_dir': os.path.join(base_dir, 'output')
        },
        'preprocessing': {
            'categorical_columns': [
                'company_size',
                'lead_source',
                'industry',
                'country'
            ],
            'numerical_columns': [
                'website_visits',
                'email_opens',
                'form_submissions',
                'time_on_site'
            ],
            'target_column': 'converted',
            'name_column': 'lead_name',
            'id_column': 'lead_id',
            'test_size': 0.2
        },
        'features': {
            'interaction_terms': True,
            'polynomial_features': False,
            'feature_selection': True
        },
        'model': {
            'default_model': 'xgboost',
            'random_state': 42,
            'cv_folds': 5
        }
    }
    
    # Create directories if they don't exist
    for path in config['paths'].values():
        os.makedirs(path, exist_ok=True)
    
    return config 