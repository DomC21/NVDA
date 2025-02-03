import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import requests
from datetime import datetime
from data_fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher()
        
    @patch('requests.get')
    def test_fetch_dark_pool_data(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"price": 300.5, "volume": 50000},
            {"price": 301.0, "volume": 30000}
        ]
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_dark_pool_data("NVDA")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
    @patch('requests.get')
    def test_fetch_option_volume_levels(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"strike": 300, "call_volume": 1000, "put_volume": 500},
            {"strike": 310, "call_volume": 800, "put_volume": 600}
        ]
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_option_volume_levels("NVDA")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
    @patch('requests.get')
    def test_fetch_historic_option_volume(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"timestamp": "2024-01-22", "volume": 1000, "premium": 150000},
            {"timestamp": "2024-01-22", "volume": 800, "premium": 120000}
        ]
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_historic_option_volume("NVDA", "2024-01-22")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
    @patch('requests.get')
    def test_fetch_greeks_exposure(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "gamma": 0.05,
            "delta": 0.6,
            "vanna": -0.02,
            "charm": 0.01
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_greeks_exposure("NVDA")
        self.assertIsInstance(result, dict)
        self.assertIn("gamma", result)
        self.assertIn("delta", result)
        
    @patch('requests.get')
    def test_fetch_market_tide(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sentiment": "bullish",
            "score": 0.75
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_market_tide()
        self.assertIsInstance(result, dict)
        self.assertIn("sentiment", result)
        
    @patch('requests.get')
    def test_fetch_option_flow(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"timestamp": "2024-01-22", "type": "call", "premium": 150000},
            {"timestamp": "2024-01-22", "type": "put", "premium": 120000}
        ]
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_option_flow("NVDA")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        
    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        result = self.fetcher.fetch_dark_pool_data("NVDA")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        
        result = self.fetcher.fetch_greeks_exposure("NVDA")
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()
