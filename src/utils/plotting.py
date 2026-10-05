import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

def plot_history(df, series_id, plots_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df['Price Date']), df['Modal_Price'], label='Modal Price', color='blue')
    plt.title(f'Price History: {series_id}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{series_id}_history.png'))
    plt.close()

def plot_train_val_test(train, val, test, series_id, plots_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(train['Price Date']), train['Modal_Price'], label='Train')
    plt.plot(pd.to_datetime(val['Price Date']), val['Modal_Price'], label='Validation')
    plt.plot(pd.to_datetime(test['Price Date']), test['Modal_Price'], label='Test')
    plt.title(f'Train / Validation / Test Split: {series_id}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{series_id}_split.png'))
    plt.close()

def plot_predictions(y_true, y_pred, model_name, horizon, series_id, plots_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label=f'Predicted ({model_name})', alpha=0.7)
    plt.title(f'{model_name} {horizon}-step Ahead Forecast: {series_id}')
    plt.xlabel('Time (Test Set)')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{series_id}_{model_name}_{horizon}step_forecast.png'))
    plt.close()
    
def plot_residuals(y_true, y_pred, model_name, horizon, series_id, plots_dir):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label='Residuals', color='red', alpha=0.7)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'{model_name} {horizon}-step Ahead Residuals: {series_id}')
    plt.xlabel('Time (Test Set)')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{series_id}_{model_name}_{horizon}step_residuals.png'))
    plt.close()
