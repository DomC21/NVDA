import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from sentiment_analyzer import SentimentAnalyzer, NewsItem

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.analyzer = SentimentAnalyzer(self.api_key)
        
    @patch('requests.get')
    def test_fetch_news_sentiment(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'feed': [
                {
                    'title': 'Test News',
                    'source': 'Test Source',
                    'time_published': '20240216T120000',
                    'overall_sentiment_score': '0.8',
                    'relevance_score': '0.9'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        news_items = self.analyzer.fetch_news_sentiment('NVDA')
        
        self.assertEqual(len(news_items), 1)
        self.assertEqual(news_items[0].title, 'Test News')
        self.assertEqual(news_items[0].sentiment_score, 0.8)
        self.assertEqual(news_items[0].relevance_score, 0.9)
        
    def test_analyze_sentiment(self):
        now = datetime.now()
        news_items = [
            NewsItem(
                title='Test 1',
                source='Source 1',
                published_at=now - timedelta(hours=1),
                sentiment_score=0.8,
                relevance_score=0.9,
                impact_score=0.72
            ),
            NewsItem(
                title='Test 2',
                source='Source 2',
                published_at=now - timedelta(hours=24),
                sentiment_score=-0.5,
                relevance_score=0.7,
                impact_score=-0.35
            )
        ]
        
        analysis = self.analyzer.analyze_sentiment(news_items)
        
        self.assertIn('composite_score', analysis)
        self.assertIn('recent_sentiment', analysis)
        self.assertIn('sentiment_momentum', analysis)
        self.assertIn('confidence', analysis)
        
        self.assertTrue(-1 <= analysis['composite_score'] <= 1)
        self.assertTrue(0 <= analysis['confidence'] <= 1)

if __name__ == '__main__':
    unittest.main()
