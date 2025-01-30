import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from data_collector import DataCollector

class PricePredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.collector = DataCollector()
        
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
        features['News_Sentiment'] = df['News_Sentiment'].fillna(method='ffill').fillna(0.5)  # Forward fill and default to neutral
        features['Sentiment_MA5'] = features['News_Sentiment'].rolling(window=5).mean().fillna(method='ffill')
        features['Sentiment_MA10'] = features['News_Sentiment'].rolling(window=10).mean().fillna(method='ffill')
        
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
        
    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='huber',
                         metrics=['mae'])
                         
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
        
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
    def predict_next_day(self, features):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        last_sequence = features[-self.sequence_length:]
        scaled_sequence = self.scaler.transform(last_sequence)
        prediction = self.model.predict(np.array([scaled_sequence]), verbose=0)
        dummy = np.zeros((prediction.shape[0], features.shape[1]))
        dummy[:, 0] = prediction[:, 0]
        return self.scaler.inverse_transform(dummy)[0][0]
        
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
        
    def backtest(self, test_size=0.2):
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
        
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        history = self.train(X_train, y_train, epochs=50)
        
        predictions = []
        actuals = []
        
        for sequence, actual in zip(X_test, y_test):
            sequence = np.expand_dims(sequence, axis=0)
            pred = self.model.predict(sequence, verbose=0)[0][0]
            
            # Create dummy arrays with same shape as training features
            dummy_pred = np.zeros((1, features.shape[1]))
            dummy_pred[0, 0] = pred
            pred_price = self.scaler.inverse_transform(dummy_pred)[0][0]
            
            dummy_actual = np.zeros((1, features.shape[1]))
            dummy_actual[0, 0] = actual
            actual_price = self.scaler.inverse_transform(dummy_actual)[0][0]
            
            if not (np.isnan(pred_price) or np.isnan(actual_price)):
                predictions.append(pred_price)
                actuals.append(actual_price)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': actuals,
            'training_history': history.history
        }
