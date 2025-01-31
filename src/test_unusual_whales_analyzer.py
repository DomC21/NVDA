import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from .unusual_whales_analyzer import UnusualWhalesAnalyzer, UnusualWhalesAnalysis

class TestUnusualWhalesAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = UnusualWhalesAnalyzer()
        
        # Sample dark pool data
        self.dark_pool_data = pd.DataFrame({
            'price': [300.0, 301.0, 300.5, 300.0],
            'volume': [150000, 50000, 200000, 75000]
        })
        
        # Sample options flow data
        self.options_flow_data = pd.DataFrame({
            'type': ['call', 'put', 'call', 'put'],
            'volume': [1000, 500, 800, 600],
            'premium': [150000, 70000, 120000, 90000]
        })
        
        # Sample market tide data
        self.market_tide_data = {
            'sentiment': 'bullish',
            'score': 0.75
        }
        
        # Sample greeks data
        self.greeks_data = {
            'gamma': 0.05,
            'delta': 0.6,
            'theta': -0.03,
            'vega': 0.02
        }
        
        # Sample volume levels data
        self.volume_levels_data = pd.DataFrame({
            'strike': [300, 305, 310],
            'volume': [1500, 1000, 800]
        })
        
    def test_analyze_dark_pool(self):
        result = self.analyzer.analyze_dark_pool(self.dark_pool_data)
        
        self.assertIn('net_flow', result)
        self.assertIn('block_trade_ratio', result)
        self.assertIn('price_impact', result)
        self.assertIn('volume_concentration', result)
        self.assertTrue(0 <= result['net_flow'] <= 1)
        
    def test_analyze_option_flow(self):
        result = self.analyzer.analyze_option_flow(self.options_flow_data)
        
        self.assertIn('call_put_ratio', result)
        self.assertIn('premium_ratio', result)
        self.assertIn('large_order_ratio', result)
        self.assertIn('bullish_flow_score', result)
        self.assertTrue(-1 <= result['bullish_flow_score'] <= 1)
        
    def test_analyze_market_tide(self):
        result = self.analyzer.analyze_market_tide(self.market_tide_data)
        
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.75)
        
    def test_analyze_greeks(self):
        result = self.analyzer.analyze_greeks(self.greeks_data)
        
        self.assertIn('net_gamma', result)
        self.assertIn('net_delta', result)
        self.assertIn('net_theta', result)
        self.assertIn('net_vega', result)
        
    def test_analyze_volume_levels(self):
        result = self.analyzer.analyze_volume_levels(self.volume_levels_data)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[300], 1500)
        
    def test_generate_signal(self):
        analysis = UnusualWhalesAnalysis(
            dark_pool_metrics={'net_flow': 0.7, 'block_trade_ratio': 0.3,
                             'price_impact': 0.02, 'volume_concentration': 0.4},
            option_flow_metrics={'call_put_ratio': 1.5, 'premium_ratio': 1.3,
                               'large_order_ratio': 0.2, 'bullish_flow_score': 0.3},
            market_tide_score=0.7,
            greek_exposure={'net_gamma': 0.05, 'net_delta': 0.6,
                          'net_theta': -0.03, 'net_vega': 0.02},
            volume_levels={300: 1500, 305: 1000, 310: 800},
            signal_strength=0.0
        )
        
        direction, strength = self.analyzer.generate_signal(analysis)
        
        self.assertIn(direction, ['buy', 'sell'])
        self.assertTrue(0 <= strength <= 1)
        
    def test_empty_data_handling(self):
        empty_df = pd.DataFrame()
        
        dark_pool_result = self.analyzer.analyze_dark_pool(empty_df)
        self.assertEqual(dark_pool_result['net_flow'], 0.0)
        
        option_flow_result = self.analyzer.analyze_option_flow(empty_df)
        self.assertEqual(option_flow_result['call_put_ratio'], 1.0)
        
        market_tide_result = self.analyzer.analyze_market_tide({})
        self.assertEqual(market_tide_result, 0.5)
        
        greeks_result = self.analyzer.analyze_greeks({})
        self.assertEqual(greeks_result['net_delta'], 0.0)
        
        volume_levels_result = self.analyzer.analyze_volume_levels(empty_df)
        self.assertEqual(len(volume_levels_result), 0)

if __name__ == '__main__':
    unittest.main()
