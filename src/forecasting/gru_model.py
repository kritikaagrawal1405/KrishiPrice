import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import time

class GRUModel:
    def __init__(self, units=32, learning_rate=0.001, dropout=0.1, epochs=50, batch_size=32, patience=5):
        self.units = units
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.train_time = 0
        
    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            GRU(self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.model = model
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.time()
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
            
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        self.train_time = time.time() - start_time
        return self
        
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()
