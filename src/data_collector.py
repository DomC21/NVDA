import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher

class DataCollector:
    def __init__(self):
        self.fetcher = DataFetcher()
        
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
                'vwap': a.vwap
            } for a in aggs])
            df.set_index('date', inplace=True)
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
    
    def _get_yfinance_data(self):
        """Fetch and process yfinance data"""
        try:
            df = self.fetcher.get_yfinance_data()
            return df
        except Exception as e:
            print(f"Error fetching yfinance data: {e}")
            return pd.DataFrame()
