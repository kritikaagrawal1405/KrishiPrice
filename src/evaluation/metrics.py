import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, MAPE, sMAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (safeguarded against division by zero since we cleaned targets <= 0, but still good practice)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    
    # sMAPE
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'sMAPE': float(smape)
    }
