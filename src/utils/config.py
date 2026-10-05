import os

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')

# Create dirs if they don't exist
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

CONFIG = {
    'seed': 42,
    
    # Series Configuration
    'primary_series': 'Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ',
    'secondary_series': [
        'West Bengal_paschim bardhaman_Asansol_Potato_Jyoti_FAQ',
        'Uttar Pradesh_sambhal_Sambhal_Potato_Desi_FAQ'
    ],
    
    # Horizons (in observation steps)
    'horizons': [1, 7],
    
    # Chronological Split
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Features
    'lag_features': [1, 3, 7, 14, 30],
    'rolling_windows': [7, 14, 30],
    
    # ML/DL Hyperparameters
    'lstm': {
        'units': 32,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.1,
        'patience': 5
    },
    'gru': {
        'units': 32,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.1,
        'patience': 5
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 4,
        'random_state': 42
    },
    'arima': {
        # Using a fixed justified small configuration for ARIMA.
        # Order (p, d, q)
        'order': (7, 1, 1) # using a weekly pattern
    }
}
