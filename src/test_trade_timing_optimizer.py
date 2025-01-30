import unittest
import pandas as pd
import numpy as np
from datetime import datetime, time
from trade_timing_optimizer import TradingTimeOptimizer

class TestTradingTimeOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = TradingTimeOptimizer()
        
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='H')
        self.test_data = pd.DataFrame({
            'Close': np.random.normal(100, 1, len(dates)),
            'Volume': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)
        
    def test_analyze_timing(self):
        results = self.optimizer.analyze_timing(self.test_data)
        
        self.assertIn('entry', results)
        self.assertIn('exit', results)
        self.assertIsInstance(results['entry']['time'], time)
        self.assertIsInstance(results['exit']['time'], time)
        self.assertTrue(0 <= results['entry']['score'] <= 1)
        self.assertTrue(0 <= results['exit']['score'] <= 1)
        
    def test_timing_recommendation(self):
        self.optimizer.analyze_timing(self.test_data)
        
        current_time = datetime.now()
        recommendation = self.optimizer.get_timing_recommendation(
            current_time, 'entry')
        
        self.assertIn('execute_now', recommendation)
        self.assertIn('confidence', recommendation)
        self.assertIn('optimal_hour', recommendation)
        self.assertIn('current_hour', recommendation)
        self.assertTrue(0 <= recommendation['confidence'] <= 1)
        
    def test_edge_cases(self):
        empty_data = pd.DataFrame()
        with self.assertRaises(Exception):
            self.optimizer.analyze_timing(empty_data)
            
        recommendation = self.optimizer.get_timing_recommendation(
            datetime.now(), 'entry')
        self.assertTrue(recommendation['execute_now'])
        self.assertEqual(recommendation['confidence'], 0.5)

if __name__ == '__main__':
    unittest.main()
