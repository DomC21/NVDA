import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from dark_pool_analyzer import DarkPoolAnalyzer, DarkPoolData

class TestDarkPoolAnalyzer(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.analyzer = DarkPoolAnalyzer(self.api_key)
        
    @patch('requests.get')
    def test_fetch_dark_pool_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'results': [
                {
                    't': 1644960000000,
                    's': 1000,
                    'p': 100.0,
                    'q': 1000,
                    'conditions': {'7'}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        trades = self.analyzer.fetch_dark_pool_data('NVDA')
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].volume, 1000)
        self.assertEqual(trades[0].price, 100.0)
        self.assertEqual(trades[0].side, 'buy')
        
    def test_analyze_dark_pool_activity(self):
        trades = [
            DarkPoolData(
                timestamp=datetime.now(),
                volume=1000,
                price=100.0,
                side='buy',
                block_size=100000
            ),
            DarkPoolData(
                timestamp=datetime.now(),
                volume=500,
                price=99.0,
                side='sell',
                block_size=49500
            )
        ]
        
        analysis = self.analyzer.analyze_dark_pool_activity(trades)
        
        self.assertIn('net_flow', analysis)
        self.assertIn('buy_pressure', analysis)
        self.assertIn('large_trade_impact', analysis)
        self.assertIn('volume_distribution', analysis)
        self.assertIn('composite_score', analysis)
        
        self.assertTrue(-1 <= analysis['net_flow'] <= 1)
        self.assertTrue(0 <= analysis['buy_pressure'] <= 1)
        self.assertTrue(-1 <= analysis['composite_score'] <= 1)
        
    def test_get_significant_levels(self):
        trades = [
            DarkPoolData(
                timestamp=datetime.now(),
                volume=1000,
                price=100.0,
                side='buy',
                block_size=100000
            ),
            DarkPoolData(
                timestamp=datetime.now(),
                volume=1500,
                price=100.0,
                side='buy',
                block_size=150000
            ),
            DarkPoolData(
                timestamp=datetime.now(),
                volume=500,
                price=99.0,
                side='sell',
                block_size=49500
            )
        ]
        
        levels = self.analyzer.get_significant_levels(trades)
        
        self.assertTrue(len(levels) > 0)
        for level in levels:
            self.assertIn('price', level)
            self.assertIn('volume', level)
            self.assertIn('strength', level)
            self.assertTrue(0 <= level['strength'] <= 1)

if __name__ == '__main__':
    unittest.main()
