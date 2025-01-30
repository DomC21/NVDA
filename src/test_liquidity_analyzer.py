import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from liquidity_analyzer import LiquidityAnalyzer

class TestLiquidityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = LiquidityAnalyzer(min_volume_threshold=1000)
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1H')
        self.price_data = pd.DataFrame({
            'high': np.random.normal(100, 1, len(dates)),
            'low': np.random.normal(99, 1, len(dates)),
            'close': np.random.normal(99.5, 0.5, len(dates))
        }, index=dates)
        
        self.volume_data = pd.DataFrame({
            'volume': np.random.normal(10000, 1000, len(dates))
        }, index=dates)
        
    def test_analyze_liquidity(self):
        result = self.analyzer.analyze_liquidity(
            self.price_data,
            self.volume_data,
            target_position_size=1000
        )
        
        self.assertIn('metrics', result)
        self.assertIn('timing_score', result)
        self.assertIn('execution_windows', result)
        self.assertIn('recommendation', result)
        
        metrics = result['metrics']
        self.assertTrue(0 <= metrics['spread'] <= 1)
        self.assertTrue(0 <= metrics['depth'] <= 1)
        self.assertTrue(-1 <= metrics['resilience'] <= 1)
        self.assertTrue(0 <= metrics['composite_score'] <= 1)
        
    def test_execution_windows(self):
        result = self.analyzer.analyze_liquidity(
            self.price_data,
            self.volume_data,
            target_position_size=1000
        )
        
        windows = result['execution_windows']
        if windows:
            window = windows[0]
            self.assertIn('start', window)
            self.assertIn('end', window)
            self.assertIn('score', window)
            self.assertTrue(0 <= window['score'] <= 1)
            
    def test_recommendation(self):
        result = self.analyzer.analyze_liquidity(
            self.price_data,
            self.volume_data,
            target_position_size=1000
        )
        
        recommendation = result['recommendation']
        self.assertIn('action', recommendation)
        self.assertIn('confidence', recommendation)
        self.assertIn('reason', recommendation)
        
        self.assertIn(recommendation['action'], ['execute', 'partial_execute', 'delay'])
        self.assertTrue(0 <= recommendation['confidence'] <= 1)
        
    def test_edge_cases(self):
        # Test with very low volume
        low_volume_data = self.volume_data.copy()
        low_volume_data['volume'] = 100
        
        result = self.analyzer.analyze_liquidity(
            self.price_data,
            low_volume_data,
            target_position_size=1000
        )
        
        self.assertEqual(result['recommendation']['action'], 'delay')
        self.assertLess(result['recommendation']['confidence'], 0.5)

if __name__ == '__main__':
    unittest.main()
