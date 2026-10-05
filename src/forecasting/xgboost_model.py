import xgboost as xgb
import time

class XGBoostModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, X, y):
        start_time = time.time()
        self.model.fit(X, y)
        self.train_time = time.time() - start_time
        return self
        
    def predict(self, X):
        return self.model.predict(X)
