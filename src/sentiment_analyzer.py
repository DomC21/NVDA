import requests
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

@dataclass
class NewsItem:
    title: str
    source: str
    published_at: datetime
    sentiment_score: float
    relevance_score: float
    impact_score: float

class SentimentAnalyzer:
    def __init__(self, alpha_vantage_key: str):
        self.api_key = alpha_vantage_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def fetch_news_sentiment(self, symbol: str) -> List[NewsItem]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.api_key,
            "limit": 50
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        news_items = []
        for item in data.get('feed', []):
            sentiment_score = float(item.get('overall_sentiment_score', 0))
            relevance_score = float(item.get('relevance_score', 0))
            impact_score = self._calculate_impact_score(sentiment_score, relevance_score)
            
            news_items.append(NewsItem(
                title=item.get('title', ''),
                source=item.get('source', ''),
                published_at=datetime.strptime(item.get('time_published', ''), '%Y%m%dT%H%M%S'),
                sentiment_score=sentiment_score,
                relevance_score=relevance_score,
                impact_score=impact_score
            ))
            
        return sorted(news_items, key=lambda x: x.published_at, reverse=True)
        
    def _calculate_impact_score(self, sentiment: float, relevance: float) -> float:
        return sentiment * relevance
        
    def analyze_sentiment(
            self,
            news_items: List[NewsItem],
            time_decay_factor: float = 0.9
    ) -> Dict[str, float]:
        if not news_items:
            return {
                'composite_score': 0.0,
                'recent_sentiment': 0.0,
                'sentiment_momentum': 0.0,
                'confidence': 0.0
            }
            
        now = datetime.now()
        weighted_scores: List[float] = []
        
        for i, item in enumerate(news_items):
            time_diff = (now - item.published_at).total_seconds() / 3600  # hours
            time_weight = time_decay_factor ** (time_diff / 24)  # decay per day
            weighted_scores.append(float(item.impact_score * time_weight))
            
        recent_sentiment = float(np.mean(weighted_scores[:5]) if len(weighted_scores) >= 5 else np.mean(weighted_scores))
        sentiment_momentum = self._calculate_sentiment_momentum(weighted_scores)
        
        composite_score = float(0.7 * recent_sentiment + 0.3 * sentiment_momentum)
        confidence = min(1.0, len(news_items) / 10)  # More news items = higher confidence
        
        return {
            'composite_score': composite_score,
            'recent_sentiment': recent_sentiment,
            'sentiment_momentum': sentiment_momentum,
            'confidence': confidence
        }
        
    def _calculate_sentiment_momentum(self, scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
            
        recent_avg = float(np.mean(scores[:len(scores)//2]))
        older_avg = float(np.mean(scores[len(scores)//2:]))
        return float(recent_avg - older_avg)
