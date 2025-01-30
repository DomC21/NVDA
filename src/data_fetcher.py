import yfinance as yf
from polygon import RESTClient
from alpha_vantage.timeseries import TimeSeries
from config import POLYGON_API_KEY, ALPHA_VANTAGE_API_KEY, SYMBOL

class DataFetcher:
    def __init__(self):
        self.polygon_client = RESTClient(POLYGON_API_KEY)
        self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY)
        self.yf_ticker = yf.Ticker(SYMBOL)

    def get_polygon_data(self):
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        return self.polygon_client.get_aggs(
            ticker=SYMBOL,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )

    def get_alpha_vantage_data(self):
        data, _ = self.alpha_vantage.get_daily(symbol=SYMBOL, outputsize='full')
        return data.head(730)

    def get_yfinance_data(self):
        return self.yf_ticker.history(period="2y")
