import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import calculate_metrics
from forecasting.baselines import NaiveModel, MovingAverageModel
from forecasting.arima_model import ARIMAModel
from forecasting.xgboost_model import XGBoostModel
from forecasting.lstm_model import LSTMModel
from forecasting.gru_model import GRUModel
from utils.config import CONFIG, MODELS_DIR, PLOTS_DIR, PREDICTIONS_DIR
from utils.leakage_checks import full_leakage_validation
from utils.plotting import plot_history, plot_train_val_test, plot_predictions, plot_residuals
import joblib

class ForecastPipeline:
    def __init__(self, data, series_id, horizon=1, experiment='A'):
        self.data = data[data['Series_ID'] == series_id].copy()
        self.series_id = series_id
        self.horizon = horizon
        self.experiment = experiment
        self.target_col = f'target_{horizon}'
        self.models = {}
        self.results = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_length = 7
        
    def _prepare_data(self):
        df = self.data.copy()
        df['Price Date'] = pd.to_datetime(df['Price Date'])
        
        if f'target_date_{self.horizon}' not in df.columns:
            # Fallback if target_date was missed
            df[f'target_date_{self.horizon}'] = df['Price Date'].shift(-self.horizon)
            
        df[f'target_date_{self.horizon}'] = pd.to_datetime(df[f'target_date_{self.horizon}'])
        
        if self.experiment == 'A':
            feature_cols = ['Modal_Price']
        else:
            feature_cols = ['Modal_Price'] + [c for c in df.columns if c.startswith('Modal_Price_lag_')] + \
                           [c for c in df.columns if c.startswith('Modal_Price_rolling_')] + \
                           ['day_of_week', 'month', 'week_of_year', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
            if 'price_spread' in df.columns:
                feature_cols += ['price_spread', 'spread_ratio']
                
        # Drop rows with NaN in features or target
        df = df.dropna(subset=[self.target_col, f'target_date_{self.horizon}'] + feature_cols).copy()
        df = df.sort_values('Price Date').reset_index(drop=True)
        
        # Horizon-aware chronological split with strict gap
        n = len(df)
        train_idx = int(n * CONFIG['train_ratio'])
        val_idx = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))
        
        train_cutoff_origin = df.iloc[train_idx]['Price Date']
        val_cutoff_origin = df.iloc[val_idx]['Price Date']
        
        self.train = df[df[f'target_date_{self.horizon}'] <= train_cutoff_origin].copy()
        self.val = df[(df['Price Date'] > train_cutoff_origin) & (df[f'target_date_{self.horizon}'] <= val_cutoff_origin)].copy()
        self.test = df[df['Price Date'] > val_cutoff_origin].copy()
        
        self.feature_cols = feature_cols
        
        # Scaling (fit strictly on train)
        self.X_train = self.scaler_X.fit_transform(self.train[feature_cols])
        self.X_val = self.scaler_X.transform(self.val[feature_cols])
        self.X_test = self.scaler_X.transform(self.test[feature_cols])
        
        self.y_train = self.train[self.target_col].values
        self.y_val = self.val[self.target_col].values
        self.y_test = self.test[self.target_col].values
        
        # Global sequences for DL boundary handling
        self.scaler_y.fit(self.train[[self.target_col]])
        
        X_all = self.scaler_X.transform(df[feature_cols])
        y_all_scaled = self.scaler_y.transform(df[[self.target_col]]).flatten()
        y_all_unscaled = df[self.target_col].values
        t_dates = df[f'target_date_{self.horizon}'].values
        
        p_dates = df['Price Date'].values
        
        Xs, ys_scaled, ys_unscaled, target_d, origin_d = [], [], [], [], []
        for i in range(len(X_all) - self.seq_length + 1):
            Xs.append(X_all[i:i + self.seq_length])
            ys_scaled.append(y_all_scaled[i + self.seq_length - 1])
            ys_unscaled.append(y_all_unscaled[i + self.seq_length - 1])
            target_d.append(t_dates[i + self.seq_length - 1])
            origin_d.append(p_dates[i + self.seq_length - 1])
            
        Xs, ys_scaled, ys_unscaled, target_d, origin_d = np.array(Xs), np.array(ys_scaled), np.array(ys_unscaled), np.array(target_d), np.array(origin_d)
        
        train_mask = target_d <= train_cutoff_origin
        val_mask = (origin_d > train_cutoff_origin) & (target_d <= val_cutoff_origin)
        test_mask = origin_d > val_cutoff_origin
        
        self.dl_data = {
            'X_train': Xs[train_mask], 'y_train': ys_scaled[train_mask],
            'X_val': Xs[val_mask], 'y_val': ys_scaled[val_mask],
            'X_test': Xs[test_mask], 'y_test': ys_unscaled[test_mask],
            'y_test_scaled': ys_scaled[test_mask],
            'target_dates_test': target_d[test_mask]
        }
        
        # Enforce robust leakage checks after all structures are created
        full_leakage_validation(self)
        
    def run_baselines(self):
        test_dates = self.test[f'target_date_{self.horizon}'].values
        naive = NaiveModel()
        preds_test = naive.predict(self.test)
        self._record_result('Naive', 'test', self.y_test, preds_test, 0, test_dates)
        
        try:
            ma = MovingAverageModel(window=7)
            preds_test = ma.predict(self.test)
            self._record_result('Moving Average', 'test', self.y_test, preds_test, 0, test_dates)
        except ValueError:
            pass

    def run_arima(self):
        test_dates = self.test[f'target_date_{self.horizon}'].values
        arima = ARIMAModel(order=CONFIG['arima']['order'])
        history_series = pd.concat([self.train['Modal_Price'], self.val['Modal_Price']])
        preds, t_time = arima.rolling_forecast(history_series, self.test['Modal_Price'], step=self.horizon)
        self._record_result('ARIMA', 'test', self.y_test, preds, t_time, test_dates, arima)

    def run_xgboost(self):
        if self.experiment == 'B':
            test_dates = self.test[f'target_date_{self.horizon}'].values
            xgb = XGBoostModel(**CONFIG['xgboost'])
            xgb.fit(self.X_train, self.y_train)
            preds = xgb.predict(self.X_test)
            self._record_result('XGBoost', 'test', self.y_test, preds, xgb.train_time, test_dates, xgb)

    def run_dl(self):
        X_tr = self.dl_data['X_train']
        y_tr = self.dl_data['y_train']
        X_v = self.dl_data['X_val']
        y_v = self.dl_data['y_val']
        X_test = self.dl_data['X_test']
        y_test_unscaled = self.dl_data['y_test']
        dl_dates = self.dl_data['target_dates_test']
        
        if len(X_tr) == 0: return
        
        # LSTM
        lstm = LSTMModel(**CONFIG['lstm'])
        lstm.fit(X_tr, y_tr, X_v, y_v)
        preds_scaled = lstm.predict(X_test)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        self._record_result('LSTM', 'test', y_test_unscaled, preds, lstm.train_time, dl_dates, lstm)
        
        # GRU
        gru = GRUModel(**CONFIG['gru'])
        gru.fit(X_tr, y_tr, X_v, y_v)
        preds_scaled = gru.predict(X_test)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        self._record_result('GRU', 'test', y_test_unscaled, preds, gru.train_time, dl_dates, gru)

    def _record_result(self, model_name, split, y_true, y_pred, train_time, target_dates=None, model_obj=None):
        metrics = calculate_metrics(y_true, y_pred)
        self.results.append({
            'Model': model_name,
            'Experiment': self.experiment,
            'Horizon': f'{self.horizon}-step',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'sMAPE': metrics['sMAPE'],
            'Training Time': train_time
        })
        
        preds_dict = {'Actual': y_true, 'Predicted': y_pred}
        if target_dates is not None:
            preds_dict['Date'] = target_dates
            
        preds_df = pd.DataFrame(preds_dict)
        preds_path = os.path.join(PREDICTIONS_DIR, f"{self.series_id}_{model_name}_Exp{self.experiment}_{self.horizon}step_{split}.csv".replace(" ", "_").replace("/", "_"))
        preds_df.to_csv(preds_path, index=False)
        
        plot_predictions(y_true, y_pred, model_name, self.horizon, self.series_id.split('_')[-1], PLOTS_DIR)
        plot_residuals(y_true, y_pred, model_name, self.horizon, self.series_id.split('_')[-1], PLOTS_DIR)
        
        if model_obj is not None:
            model_path = os.path.join(MODELS_DIR, f"{self.series_id}_{model_name}_Exp{self.experiment}_{self.horizon}step".replace(" ", "_").replace("/", "_"))
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'save'):
                model_obj.model.save(model_path + '.keras')
            else:
                joblib.dump(model_obj, model_path + '.pkl')
        
    def execute(self):
        self._prepare_data()
        self.run_baselines()
        if self.experiment == 'A':
            self.run_arima()
            self.run_dl()
        if self.experiment == 'B':
            self.run_xgboost()
            self.run_dl()
            
        return pd.DataFrame(self.results)
