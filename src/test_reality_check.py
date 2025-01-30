import unittest
import pandas as pd
import numpy as np
from reality_check import RealityCheck

class TestRealityCheck(unittest.TestCase):
    def setUp(self):
        self.checker = RealityCheck(
            min_volume_threshold=100000,
            max_spread_threshold=0.03,
            min_confidence=0.6
        )
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        self.market_data = pd.DataFrame({
            'Close': 100 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'High': 101 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'Low': 99 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'Volume': np.random.normal(500000, 50000, len(dates))
        }, index=dates)
        
        self.options_data = pd.DataFrame({
            'call_volume': np.random.normal(10000, 1000, len(dates)),
            'put_volume': np.random.normal(8000, 800, len(dates))
        }, index=dates)
        
    def test_volume_check(self):
        result = self.checker._check_volume(self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('volume_ratio', result['context'])
        
    def test_spread_check(self):
        result = self.checker._check_spread(self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('spread', result['context'])
        
    def test_trend_alignment(self):
        signal = {'direction': 'buy'}
        result = self.checker._check_trend_alignment(signal, self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('strength', result['context'])
        
    def test_options_flow(self):
        signal = {'direction': 'buy'}
        result = self.checker._check_options_flow(signal, self.options_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('call_put_ratio', result['context'])
        
    def test_volatility_check(self):
        result = self.checker._check_volatility(self.market_data)
        self.assertIn('valid', result)
        self.assertIn('reason', result)
        self.assertIn('context', result)
        self.assertIn('volatility', result['context'])
        
    def test_signal_validation(self):
        signal = {'direction': 'buy'}
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            self.options_data
        )
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIsInstance(result.failure_reasons, list)
        self.assertIsInstance(result.market_context, dict)
        
    def test_edge_cases(self):
        # Test with low volume
        low_volume_data = self.market_data.copy()
        low_volume_data['Volume'] = 1000
        
        result = self.checker.validate_signal(
            {'direction': 'buy'},
            low_volume_data
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(len(result.failure_reasons) > 0)
        
        # Test with high spread
        high_spread_data = self.market_data.copy()
        high_spread_data['High'] *= 1.1
        high_spread_data['Low'] *= 0.9
        
        result = self.checker.validate_signal(
            {'direction': 'buy'},
            high_spread_data
        )
        
        self.assertFalse(result.is_valid)

if __name__ == '__main__':
    unittest.main()
