import unittest
import pandas as pd
import numpy as np
from reality_check import RealityCheck

class TestRealityCheck(unittest.TestCase):
    def setUp(self):
        self.checker = RealityCheck(
            min_volume_threshold=100000,
            max_spread_threshold=0.03,
            min_confidence=0.6,
            max_gamma_exposure=0.01,
            max_vega_exposure=0.02,
            max_theta_decay=0.005,
            min_delta_hedge=0.3,
            high_block_threshold=1_000_000
        )
        
        # Sample Greeks data
        self.normal_greeks = {
            'gamma': 0.002,
            'vega': 0.15,
            'theta': -0.05,
            'delta': 0.6,
            'hedge_ratio': 0.8
        }
        
        self.high_risk_greeks = {
            'gamma': 0.05,
            'vega': 0.4,
            'theta': -0.2,
            'delta': 0.9,
            'hedge_ratio': 0.2
        }
        
        # Sample dark pool data
        self.dark_pool_data = pd.DataFrame({
            'price': [100.0, 101.0, 99.0],
            'volume': [500000, 300000, 200000]
        })
        
        self.large_block_data = pd.DataFrame({
            'price': [98.0, 97.5, 97.0],
            'volume': [2000000, 1500000, 1000000]
        })
        
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
        
        # Test with all data sources
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            self.options_data,
            self.dark_pool_data,
            self.normal_greeks
        )
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIsInstance(result.failure_reasons, list)
        self.assertIsInstance(result.market_context, dict)
        self.assertIn('gamma_exposure', result.market_context)
        self.assertIn('block_sum', result.market_context)
        
        # Test with high risk conditions
        high_risk_result = self.checker.validate_signal(
            signal,
            self.market_data,
            self.options_data,
            self.large_block_data,
            self.high_risk_greeks
        )
        
        self.assertFalse(high_risk_result.is_valid)
        self.assertLess(high_risk_result.confidence, 0.5)
        self.assertTrue(len(high_risk_result.failure_reasons) > 0)
        self.assertTrue(any('gamma' in reason.lower() for reason in high_risk_result.failure_reasons))
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIsInstance(result.failure_reasons, list)
        self.assertIsInstance(result.market_context, dict)
        
    def test_dark_pool_validation(self):
        signal = {'direction': 'buy'}
        
        # Test with normal dark pool data
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            dark_pool_data=self.dark_pool_data
        )
        
        self.assertTrue(result.is_valid)
        self.assertIn('block_sum', result.market_context)
        
        # Test with large block data
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            dark_pool_data=self.large_block_data
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('dark pool blocks' in reason.lower() for reason in result.failure_reasons))
        
    def test_greeks_validation(self):
        signal = {'direction': 'buy'}
        
        # Test with normal Greeks
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            greeks_data=self.normal_greeks
        )
        
        self.assertTrue(result.is_valid)
        self.assertIn('gamma_exposure', result.market_context)
        self.assertIn('vega_exposure', result.market_context)
        self.assertIn('theta_decay', result.market_context)
        
        # Test with high risk Greeks
        result = self.checker.validate_signal(
            signal,
            self.market_data,
            greeks_data=self.high_risk_greeks
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any('gamma' in reason.lower() for reason in result.failure_reasons))
        self.assertTrue(any('vega' in reason.lower() for reason in result.failure_reasons))
        
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
