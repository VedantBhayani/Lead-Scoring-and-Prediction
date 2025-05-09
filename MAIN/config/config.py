from typing import Dict, Any

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'target_column': 'converted',
    'date_columns': ['created_at', 'last_activity_date'],
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
    ]
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'date_columns': ['created_at', 'last_activity_date'],
    'interaction_pairs': [
        ('company_size_score', 'lead_source_score'),
        ('engagement_score', 'company_size_score')
    ],
    'aggregation_config': {
        'engagement_weights': {
            'website_visits': 0.4,
            'email_opens': 0.3,
            'form_submissions': 0.3
        },
        'company_size_mapping': {
            'Small': 1,
            'Medium': 2,
            'Large': 3,
            '1-10': 1,
            '11-50': 2,
            '51-200': 3,
            '201-500': 4,
            '501-1000': 5,
            '1001-5000': 6,
            '5001-10000': 7,
            '10001+': 8
        },
        'lead_source_mapping': {
            'Website': 1,
            'Referral': 2,
            'Social Media': 3,
            'Email': 4,
            'Email Campaign': 4,
            'Trade Show': 5
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    'default_model': 'xgboost',
    'random_forest': {
        'n_estimators': 1000,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'min_child_weight': 3,
        'subsample': 0.8
    },
    'lightgbm': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'num_leaves': 50,
        'feature_fraction': 0.8
    },
    'scoring': {
        'hot_threshold': 80,
        'warm_threshold': 50,
        'cold_threshold': 0
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'plot_style': 'seaborn',
    'color_palette': 'husl',
    'figure_size': (12, 6),
    'dpi': 100
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': "Lead Scoring Dashboard",
    'page_icon': "📊",
    'layout': "wide",
    'navigation_items': [
        "Lead Scoring",
        "Model Performance",
        "Feature Analysis",
        "Trends"
    ]
}

# File paths
PATH_CONFIG = {
    'data_dir': 'data',
    'models_dir': 'models',
    'results_dir': 'results',
    'visualizations_dir': 'visualizations'
}

# Add these functions to provide default configuration
def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the lead scoring system."""
    return {
        'preprocessing': PREPROCESSING_CONFIG,
        'features': FEATURE_CONFIG,
        'model': MODEL_CONFIG,
        'paths': {
            'data_dir': 'data',
            'model_dir': 'models',
            'output_dir': 'output',
            'results_dir': 'results'
        }
    }

def get_config() -> Dict[str, Any]:
    """Get configuration."""
    try:
        # Here you would typically load config from file or environment
        # For now, we just return the default config
        return get_default_config()
    except Exception as e:
        print(f"Error loading config: {str(e)}. Using default config.")
        return get_default_config() 