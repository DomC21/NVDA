import yfinance as yf
import pandas as pd
from polygon import RESTClient
from alpha_vantage.timeseries import TimeSeries
from config import POLYGON_API_KEY, ALPHA_VANTAGE_API_KEY, SYMBOL
import requests
from datetime import datetime, timedelta
import json

class DataFetcher:
    def __init__(self):
        self.polygon_client = RESTClient(POLYGON_API_KEY)
        self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY)
        self.yf_ticker = yf.Ticker(SYMBOL)
        self.spy_ticker = yf.Ticker("SPY")
        self.soxx_ticker = yf.Ticker("SOXX")
        self.sentiment_cache = {}

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
        nvda_data = self.yf_ticker.history(period="2y")
        spy_data = self.spy_ticker.history(period="2y")
        soxx_data = self.soxx_ticker.history(period="2y")
        
        # Calculate correlations and relative strength
        nvda_returns = nvda_data['Close'].pct_change()
        spy_returns = spy_data['Close'].pct_change()
        soxx_returns = soxx_data['Close'].pct_change()
        
        nvda_data['SPY_Correlation'] = nvda_returns.rolling(window=20).corr(spy_returns)
        nvda_data['SOXX_Correlation'] = nvda_returns.rolling(window=20).corr(soxx_returns)
        nvda_data['Market_RS'] = (nvda_returns + 1).cumprod() / (spy_returns + 1).cumprod()
        nvda_data['Sector_RS'] = (nvda_returns + 1).cumprod() / (soxx_returns + 1).cumprod()
        
        # Add sentiment data
        try:
            sentiment_data = self._get_polygon_news_sentiment()
            if sentiment_data:
                sentiment_series = pd.Series(sentiment_data)
                sentiment_series.index = pd.to_datetime(sentiment_series.index)
                nvda_data['News_Sentiment'] = sentiment_series.reindex(nvda_data.index).fillna(0.5)
            else:
                nvda_data['News_Sentiment'] = pd.Series(0.5, index=nvda_data.index)
        except Exception as e:
            print(f"Error processing sentiment data: {e}")
            nvda_data['News_Sentiment'] = pd.Series(0.5, index=nvda_data.index)
        
        return nvda_data
        
    def _get_polygon_news_sentiment(self):
        sentiment_scores = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        try:
            news = self.polygon_client.list_ticker_news(
                ticker=SYMBOL,
                published_utc_gte=start_date.strftime("%Y-%m-%d"),
                published_utc_lte=end_date.strftime("%Y-%m-%d"),
                limit=1000
            )
            
            for article in news:
                date = datetime.strptime(article.published_utc[:10], "%Y-%m-%d")
                sentiment = article.sentiment_score if hasattr(article, 'sentiment_score') else 0.5
                
                if date not in sentiment_scores:
                    sentiment_scores[date] = []
                sentiment_scores[date].append(sentiment)
            
            # Average sentiment scores for each day
            daily_sentiment = {date: sum(scores)/len(scores) for date, scores in sentiment_scores.items()}
            
            return daily_sentiment
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return {}
