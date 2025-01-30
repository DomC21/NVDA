import unittest
import pandas as pd
import numpy as np
from volume_analyzer import VolumeAnalyzer

class TestVolumeAnalyzer(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        self.analyzer = VolumeAnalyzer(self.test_data)
        
    def test_calculate_volume_profile(self):
        profile = self.analyzer.calculate_volume_profile()
        
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile.price_levels), 49)
        self.assertEqual(len(profile.volume_at_price), 49)
        self.assertTrue(profile.value_area_high > profile.value_area_low)
        self.assertTrue(profile.poc_price >= profile.value_area_low)
        self.assertTrue(profile.poc_price <= profile.value_area_high)
        
    def test_get_support_resistance(self):
        levels = self.analyzer.get_support_resistance()
        
        self.assertIsInstance(levels, list)
        if levels:
            self.assertTrue(all(isinstance(level, float) for level in levels))
            
    def test_analyze_volume_trend(self):
        trend = self.analyzer.analyze_volume_trend()
        
        self.assertIn('volume_ratio', trend)
        self.assertIn('volume_momentum', trend)
        self.assertIn('trend_strength', trend)
        self.assertIsInstance(trend['volume_ratio'], float)
        self.assertIsInstance(trend['volume_momentum'], float)
        self.assertIsInstance(trend['trend_strength'], float)
        
    def test_get_entry_exit_signals(self):
        current_price = self.test_data['Close'].iloc[-1]
        signals = self.analyzer.get_entry_exit_signals(current_price)
        
        self.assertIn('long', signals)
        self.assertIn('short', signals)
        
        for direction in ['long', 'short']:
            self.assertIn('entry_price', signals[direction])
            self.assertIn('stop_loss', signals[direction])
            self.assertIn('take_profit', signals[direction])
            self.assertIn('risk_reward_ratio', signals[direction])
            self.assertIn('volume_confidence', signals[direction])

if __name__ == '__main__':
    unittest.main()
