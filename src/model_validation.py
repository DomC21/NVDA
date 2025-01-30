import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

class TimeSeriesValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, data, sequence_length=60):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 0])
        return np.array(X), np.array(y)
    
    def create_lstm_model(self, input_shape, units=[128, 64, 32], dropout_rate=0.3):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units[0], return_sequences=True, input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units[1], return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units[2]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def create_gru_model(self, input_shape, units=[128, 64, 32], dropout_rate=0.3):
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(units[0], return_sequences=True, input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.GRU(units[1], return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.GRU(units[2]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])
        return model
    
    def get_callbacks(self):
        return [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
        ]
    
    def train_neural_network(self, model, X_train, y_train, X_val, y_val, batch_size=64):
        callbacks = self.get_callbacks()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return history
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, params=None):
        if params is None:
            params = {
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.9
            }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train.reshape(X_train.shape[0], -1),
            y_train,
            eval_set=[(X_val.reshape(X_val.shape[0], -1), y_val)],
            early_stopping_rounds=20,
            verbose=0
        )
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val, params=None):
        if params is None:
            params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_split': 5
            }
        model = GradientBoostingRegressor(**params)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        return model
    
    def evaluate_fold(self, models, X_test, y_test):
        predictions = {}
        metrics = {}
        
        for name, model in models.items():
            if isinstance(model, (xgb.XGBRegressor, GradientBoostingRegressor)):
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                pred = model.predict(X_test_reshaped)
            else:
                pred = model.predict(X_test)
            
            predictions[name] = pred
            metrics[name] = {
                'mae': mean_absolute_error(y_test, pred),
                'mape': np.mean(np.abs((y_test - pred) / y_test)) * 100,
                'rmse': np.sqrt(mean_squared_error(y_test, pred))
            }
        
        return predictions, metrics
    
    def ensemble_predict(self, models, X):
        predictions = {}
        for name, model in models.items():
            if isinstance(model, (xgb.XGBRegressor, GradientBoostingRegressor)):
                X_reshaped = X.reshape(X.shape[0], -1)
                pred = model.predict(X_reshaped)
            else:
                pred = model.predict(X)
            predictions[name] = pred
        
        weights = {
            'lstm': 0.3,
            'gru': 0.3,
            'xgboost': 0.2,
            'gradient_boosting': 0.2
        }
        
        ensemble_pred = np.zeros_like(predictions['lstm'])
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
            
        return ensemble_pred, predictions
    
    def cross_validate(self, X, y, hyperparameters=None):
        results = {
            'fold_metrics': [],
            'ensemble_metrics': [],
            'predictions': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            val_size = len(X_test)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]
            
            models = {}
            
            lstm_model = self.create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            lstm_model.compile(optimizer='adam', loss='huber')
            self.train_neural_network(lstm_model, X_train, y_train, X_val, y_val)
            models['lstm'] = lstm_model
            
            gru_model = self.create_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            gru_model.compile(optimizer='adam', loss='huber')
            self.train_neural_network(gru_model, X_train, y_train, X_val, y_val)
            models['gru'] = gru_model
            
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
            models['xgboost'] = xgb_model
            
            gb_model = self.train_gradient_boosting(X_train, y_train, X_val, y_val)
            models['gradient_boosting'] = gb_model
            
            predictions, metrics = self.evaluate_fold(models, X_test, y_test)
            ensemble_pred, individual_preds = self.ensemble_predict(models, X_test)
            
            ensemble_metrics = {
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'mape': np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100,
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred))
            }
            
            results['fold_metrics'].append(metrics)
            results['ensemble_metrics'].append(ensemble_metrics)
            results['predictions'].append({
                'true': y_test,
                'ensemble': ensemble_pred,
                'individual': individual_preds
            })
        
        return results
