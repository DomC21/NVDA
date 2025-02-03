import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
import numpy as np

class DataCollector:
    def __init__(self):
        self.fetcher = DataFetcher()
        
    def _calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_stochastic(self, df, period=14):
        lowest_low = df['low'].rolling(period).min()
        highest_high = df['high'].rolling(period).max()
        k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(3).mean()
        return k, d
    
    def _calculate_obv(self, df):
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_cmf(self, df, period=20):
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        cmf = mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        return cmf
    
    def _calculate_adx(self, df, period=14):
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = self._calculate_atr(df, period=1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx
    
    def collect_all_data(self):
        """Collect and combine data from all sources for NVDA"""
        # Get data from different sources
        polygon_data = self._get_polygon_data()
        alpha_vantage_data = self._get_alpha_vantage_data()
        yfinance_data = self._get_yfinance_data()
        
        # Validate data length
        min_required_days = 500  # ~2 years of trading days
        data_lengths = {
            'polygon': len(polygon_data),
            'alpha_vantage': len(alpha_vantage_data),
            'yfinance': len(yfinance_data)
        }
        
        for source, length in data_lengths.items():
            if length < min_required_days:
                print(f"Warning: {source} data has insufficient history. Got {length} days, expected {min_required_days}")
        
        return {
            'polygon': polygon_data,
            'alpha_vantage': alpha_vantage_data,
            'yfinance': yfinance_data
        }
    
    def _get_polygon_data(self):
        """Fetch and process Polygon.io data"""
        try:
            aggs = self.fetcher.get_polygon_data()
            df = pd.DataFrame([{
                'date': datetime.fromtimestamp(a.timestamp/1000),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume,
                'vwap': a.vwap,
                'transactions': a.transactions if hasattr(a, 'transactions') else None
            } for a in aggs])
            df.set_index('date', inplace=True)
            
            # Calculate additional technical indicators
            df['Daily_Return'] = df['close'].pct_change()
            df['Rolling_Volatility'] = df['Daily_Return'].rolling(20).std()
            df['ATR'] = self._calculate_atr(df, period=14)
            df['Stochastic_K'], df['Stochastic_D'] = self._calculate_stochastic(df, period=14)
            df['OBV'] = self._calculate_obv(df)
            df['CMF'] = self._calculate_cmf(df, period=20)
            df['ADX'] = self._calculate_adx(df, period=14)
            
            return df
        except Exception as e:
            print(f"Error fetching Polygon data: {e}")
            return pd.DataFrame()
    
    def _get_alpha_vantage_data(self):
        """Fetch and process Alpha Vantage data"""
        try:
            data = self.fetcher.get_alpha_vantage_data()
            df = pd.DataFrame(data).T
            df.index = pd.to_datetime(df.index)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            return df
        except Exception as e:
            print(f"Error fetching Alpha Vantage data: {e}")
            return pd.DataFrame()
    
    def _get_yfinance_data(self) -> pd.DataFrame:
        """Fetch and process yfinance data"""
        try:
            df = self.fetcher.get_yfinance_data()
            if df is None or df.empty:
                return pd.DataFrame()
                
            # Add derivative features
            df['Price_Range'] = df['High'] - df['Low']
            df['Gap'] = df['Open'] - df['Close'].shift(1)
            df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            df['Volume_Intensity'] = df['Volume'] * df['Daily_Return'].abs()
            df['Volume_Force'] = df['Volume'] * df['Daily_Return']
            df['Volatility_Ratio'] = (
                df['Rolling_Volatility'] / df['Rolling_Volatility'].rolling(100).mean()
            )
            
            return df
        except Exception as e:
            print(f"Error fetching yfinance data: {e}")
            return pd.DataFrame()
            
    def collect_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch market data for a given symbol"""
        try:
            df = self.fetcher.get_yfinance_data()
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
            
    def collect_historical_data(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """Collect historical data for a given symbol and date range."""
        try:
            df = self.fetcher.get_yfinance_data(start_date, end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error collecting historical data: {e}")
            return pd.DataFrame()
