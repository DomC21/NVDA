import unittest
import pandas as pd
import numpy as np
from market_regime import MarketRegimeDetector, MarketRegime

class TestMarketRegimeDetector(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start='2023-01-01', periods=100)
        
        trend_up = np.linspace(100, 150, 100) + np.random.normal(0, 1, 100)
        self.trending_up_data = pd.DataFrame({
            'Close': trend_up
        }, index=dates)
        
        trend_down = np.linspace(150, 100, 100) + np.random.normal(0, 1, 100)
        self.trending_down_data = pd.DataFrame({
            'Close': trend_down
        }, index=dates)
        
        ranging = 100 + np.random.normal(0, 2, 100)
        self.ranging_data = pd.DataFrame({
            'Close': ranging
        }, index=dates)
        
        high_vol = 100 + np.random.normal(0, 5, 100)
        self.high_volatility_data = pd.DataFrame({
            'Close': high_vol
        }, index=dates)
        
    def test_trending_up_detection(self):
        detector = MarketRegimeDetector(self.trending_up_data)
        regime = detector.detect_regime()
        self.assertEqual(regime.regime, MarketRegime.TRENDING_UP)
        self.assertGreater(regime.confidence, 0.6)
        
    def test_trending_down_detection(self):
        detector = MarketRegimeDetector(self.trending_down_data)
        regime = detector.detect_regime()
        self.assertEqual(regime.regime, MarketRegime.TRENDING_DOWN)
        self.assertGreater(regime.confidence, 0.6)
        
    def test_ranging_detection(self):
        detector = MarketRegimeDetector(self.ranging_data)
        regime = detector.detect_regime()
        self.assertEqual(regime.regime, MarketRegime.RANGING)
        self.assertGreater(regime.confidence, 0.5)
        
    def test_high_volatility_detection(self):
        detector = MarketRegimeDetector(self.high_volatility_data)
        regime = detector.detect_regime()
        self.assertEqual(regime.regime, MarketRegime.HIGH_VOLATILITY)
        self.assertGreater(regime.confidence, 0.7)
        
    def test_regime_parameters(self):
        detector = MarketRegimeDetector(self.trending_up_data)
        params = detector.get_regime_parameters()
        
        self.assertIn('current_regime', params)
        self.assertIn('confidence', params)
        self.assertIn('position_size_factor', params)
        self.assertIn('stop_loss_factor', params)
        self.assertIn('take_profit_factor', params)
        
        self.assertGreater(params['confidence'], 0)
        self.assertGreater(params['position_size_factor'], 0)
        self.assertGreater(params['stop_loss_factor'], 0)
        self.assertGreater(params['take_profit_factor'], 0)

if __name__ == '__main__':
    unittest.main()
