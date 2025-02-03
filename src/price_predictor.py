import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import itertools
import os
from data_collector import DataCollector

class PricePredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.lstm_model = None
        self.gru_model = None
        self.xgb_model = None
        self.gb_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.collector = DataCollector()
        self.best_params = None
        self.cv_results = None
        
    def prepare_data(self):
        raw_data = self.collector.collect_all_data()
        if 'yfinance' not in raw_data or raw_data['yfinance'] is None:
            raise ValueError("Failed to fetch YFinance data")
            
        df = raw_data['yfinance']
        features = self._extract_features(df)
        return self._create_sequences(features)
        
    def _extract_features(self, df):
        features = pd.DataFrame()
        
        # Price and volume features
        features['Close'] = df['Close']
        features['Volume'] = df['Volume']
        features['RSI'] = self._calculate_rsi(df['Close'])
        features['SMA_20'] = df['Close'].rolling(window=20).mean()
        features['SMA_50'] = df['Close'].rolling(window=50).mean()
        features['MACD'] = self._calculate_macd(df['Close'])
        
        # Market context features
        features['SPY_Correlation'] = df['SPY_Correlation']
        features['SOXX_Correlation'] = df['SOXX_Correlation']
        features['Market_RS'] = df['Market_RS']
        features['Sector_RS'] = df['Sector_RS']
        
        # Volatility features
        features['Daily_Return'] = df['Close'].pct_change()
        features['Volatility'] = features['Daily_Return'].rolling(window=20).std()
        
        # Sentiment features
        features['News_Sentiment'] = df['News_Sentiment'].fillna(0.5)
        features['Sentiment_MA5'] = features['News_Sentiment'].rolling(window=5).mean().bfill().fillna(0.5)
        features['Sentiment_MA10'] = features['News_Sentiment'].rolling(window=10).mean().bfill().fillna(0.5)
        
        features = features.dropna()
        return features
        
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _create_sequences(self, features):
        scaled_features = self.scaler.fit_transform(features)
        X, y = [], []
        
        for i in range(len(scaled_features) - self.sequence_length - 1):
            sequence = scaled_features[i:(i + self.sequence_length)]
            target = scaled_features[i + self.sequence_length, 0]
            
            if len(sequence) == self.sequence_length:
                X.append(sequence)
                y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Not enough data to create sequences")
            
        return X, y
        
    def build_lstm_model(self, input_shape, params=None):
        if params is None:
            params = {
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'dense_units': 16
            }
        
        model = Sequential([
            LSTM(params['lstm_units'][0], return_sequences=True, input_shape=input_shape),
            Dropout(params['dropout_rate']),
            LSTM(params['lstm_units'][1], return_sequences=True),
            Dropout(params['dropout_rate']),
            LSTM(params['lstm_units'][2], return_sequences=False),
            Dropout(params['dropout_rate']),
            Dense(params['dense_units'], activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                    loss='huber',
                    metrics=['mae'])
        
        return model
        
    def build_gru_model(self, input_shape, params=None):
        if params is None:
            params = {
                'gru_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'dense_units': 16
            }
        
        model = Sequential([
            GRU(params['gru_units'][0], return_sequences=True, input_shape=input_shape),
            Dropout(params['dropout_rate']),
            GRU(params['gru_units'][1], return_sequences=True),
            Dropout(params['dropout_rate']),
            GRU(params['gru_units'][2], return_sequences=False),
            Dropout(params['dropout_rate']),
            Dense(params['dense_units'], activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                    loss='huber',
                    metrics=['mae'])
        
        return model
        
    def build_tree_models(self, X_flat, y):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.xgb_model.fit(X_flat, y)
        self.gb_model.fit(X_flat, y)
        
    def flatten_sequences(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1] * X.shape[2]
        return X.reshape((n_samples, n_features))
        
    def grid_search_cv(self, X, y, param_grid, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_score = float('inf')
        best_params = None
        cv_results = []
        
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        for params in param_combinations:
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]), params=params)
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
                
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                val_pred = model.predict(X_val, verbose=0)
                mae = mean_absolute_error(y_val, val_pred)
                fold_scores.append(mae)
            
            avg_score = np.mean(fold_scores)
            cv_results.append({
                'params': params,
                'mean_mae': avg_score,
                'std_mae': np.std(fold_scores)
            })
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
        
        self.best_params = best_params
        self.cv_results = cv_results
        return best_params, cv_results
                         
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        # Grid search parameters for neural networks
        lstm_param_grid = {
            'lstm_units': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'dense_units': [16, 32, 64]
        }
        
        gru_param_grid = {
            'gru_units': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'dense_units': [16, 32, 64]
        }
        
        # Train LSTM model
        lstm_best_params, lstm_cv_results = self.grid_search_cv(X_train, y_train, lstm_param_grid)
        self.lstm_model = self.build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            params=lstm_best_params
        )
        
        # Train GRU model
        gru_best_params, gru_cv_results = self.grid_search_cv(X_train, y_train, gru_param_grid)
        self.gru_model = self.build_gru_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            params=gru_best_params
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath='lstm_model.h5', monitor='val_loss', save_best_only=True, mode='min')
        ]
        
        # Final training of neural networks
        self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        callbacks[1] = ModelCheckpoint(filepath='gru_model.h5', monitor='val_loss', save_best_only=True, mode='min')
        self.gru_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train tree-based models on flattened sequences
        X_flat = self.flatten_sequences(X_train)
        self.build_tree_models(X_flat, y_train)
        
        # Store best parameters
        self.best_params = {
            'lstm': lstm_best_params,
            'gru': gru_best_params
        }
        
        # Store cross-validation results
        self.cv_results = {
            'lstm': lstm_cv_results,
            'gru': gru_cv_results
        }
        
        return {
            'lstm_history': self.lstm_model.history.history,
            'gru_history': self.gru_model.history.history
        }
        
    def predict_next_day(self, features):
        try:
            if any(model is None for model in [self.lstm_model, self.gru_model, self.xgb_model, self.gb_model]):
                raise ValueError("Models not built. Call train() first.")
            
            last_sequence = features[-self.sequence_length:]
            scaled_sequence = self.scaler.transform(last_sequence)
            sequence_array = np.array([scaled_sequence])
            flat_sequence = self.flatten_sequences(sequence_array)
            
            # Get predictions from all models
            predictions = {}
            
            if self.lstm_model is not None:
                predictions['lstm'] = self.lstm_model.predict(sequence_array, verbose=0)
            else:
                raise ValueError("LSTM model not initialized")
                
            if self.gru_model is not None:
                predictions['gru'] = self.gru_model.predict(sequence_array, verbose=0)
            else:
                raise ValueError("GRU model not initialized")
                
            if self.xgb_model is not None:
                predictions['xgb'] = self.xgb_model.predict(flat_sequence).reshape(-1, 1)
            else:
                raise ValueError("XGBoost model not initialized")
                
            if self.gb_model is not None:
                predictions['gb'] = self.gb_model.predict(flat_sequence).reshape(-1, 1)
            else:
                raise ValueError("Gradient Boosting model not initialized")
        
        # Ensemble prediction (weighted average)
            weights = {'lstm': 0.3, 'gru': 0.3, 'xgb': 0.2, 'gb': 0.2}
            ensemble_pred = sum(weights[k] * v for k, v in predictions.items())
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")
        
        # Convert ensemble prediction to proper shape
        ensemble_pred_array = np.array(ensemble_pred).reshape(-1, 1)
        dummy = np.zeros((ensemble_pred_array.shape[0], features.shape[1]))
        dummy[:, 0] = ensemble_pred_array.flatten()
        return float(self.scaler.inverse_transform(dummy)[0][0])
        
    def generate_price_ranges(self, current_price, prediction):
        volatility = 0.02  # 2% assumed volatility
        return {
            'prediction': prediction,
            'confidence_ranges': {
                '90%': (prediction * (1 - volatility), prediction * (1 + volatility)),
                '70%': (prediction * (1 - volatility * 0.7), prediction * (1 + volatility * 0.7)),
                '50%': (prediction * (1 - volatility * 0.5), prediction * (1 + volatility * 0.5))
            }
        }
        
    def backtest(self, test_size=0.2, n_splits=5):
        try:
            raw_data = self.collector.collect_all_data()
            if not raw_data or 'yfinance' not in raw_data:
                raise ValueError("Failed to collect YFinance data")
            
            df = raw_data['yfinance']
            features = self._extract_features(df)
            X, y = self._create_sequences(features)
            
            # Split into train/test
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train all models
            history = self.train(X_train, y_train, epochs=50)
            
            # Prepare test data for tree models
            X_test_flat = self.flatten_sequences(X_test)
            
            # Get predictions from all models
            predictions = {}
            
            if self.lstm_model is not None:
                predictions['lstm'] = self.lstm_model.predict(X_test, verbose=0)
            if self.gru_model is not None:
                predictions['gru'] = self.gru_model.predict(X_test, verbose=0)
            if self.xgb_model is not None:
                predictions['xgb'] = self.xgb_model.predict(X_test_flat).reshape(-1, 1)
            if self.gb_model is not None:
                predictions['gb'] = self.gb_model.predict(X_test_flat).reshape(-1, 1)
        
            # Calculate ensemble prediction
            weights = {'lstm': 0.3, 'gru': 0.3, 'xgb': 0.2, 'gb': 0.2}
            ensemble_pred = sum(weights[k] * v for k, v in predictions.items())
        except Exception as e:
            raise ValueError(f"Error during backtesting: {str(e)}")
        
        # Calculate metrics for each model
        def calculate_metrics(predictions, actuals):
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            return {'mae': mae, 'mape': mape, 'rmse': rmse}
        
        # Calculate metrics for each model's predictions
        metrics = {
            'lstm': calculate_metrics(predictions['lstm'], y_test),
            'gru': calculate_metrics(predictions['gru'], y_test),
            'xgboost': calculate_metrics(predictions['xgb'], y_test),
            'gradient_boosting': calculate_metrics(predictions['gb'], y_test),
            'ensemble': calculate_metrics(ensemble_pred, y_test)
        }
        
        # Create dummy arrays for inverse transformation
        ensemble_pred_array = np.array(ensemble_pred)
        if len(ensemble_pred_array.shape) == 1:
            ensemble_pred_array = ensemble_pred_array.reshape(-1, 1)
            
        dummy = np.zeros((ensemble_pred_array.shape[0], features.shape[1]))
        
        # Transform predictions back to original scale
        def inverse_transform_preds(preds):
            preds_array = np.array(preds)
            if len(preds_array.shape) == 1:
                preds_array = preds_array.reshape(-1, 1)
            dummy[:, 0] = preds_array.flatten()
            return self.scaler.inverse_transform(dummy)[:, 0]
        
        return {
            'metrics': metrics,
            'predictions': {
                'lstm': inverse_transform_preds(predictions['lstm']),
                'gru': inverse_transform_preds(predictions['gru']),
                'xgboost': inverse_transform_preds(predictions['xgb']),
                'gradient_boosting': inverse_transform_preds(predictions['gb']),
                'ensemble': inverse_transform_preds(ensemble_pred)
            },
            'actuals': inverse_transform_preds(y_test.reshape(-1, 1)),
            'training_history': history
        }
