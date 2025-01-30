import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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
        features['Close'] = df['Close']
        features['Volume'] = df['Volume']
        features['RSI'] = self._calculate_rsi(df['Close'])
        features['SMA_20'] = df['Close'].rolling(window=20).mean()
        features['SMA_50'] = df['Close'].rolling(window=50).mean()
        features['MACD'] = self._calculate_macd(df['Close'])
        
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
        
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_features[i + self.sequence_length, 0])
            
        return np.array(X), np.array(y)
        
    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
                         
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
    def predict_next_day(self, features):
        last_sequence = features[-self.sequence_length:]
        scaled_sequence = self.scaler.transform(last_sequence)
        prediction = self.model.predict(np.array([scaled_sequence]))
        return self.scaler.inverse_transform(prediction)[0][0]
        
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
