import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
from .market_regime_analyzer import MarketRegimeAnalyzer, MarketRegimeAnalysis

class TestMarketRegimeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = MarketRegimeAnalyzer()
        
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.price_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.normal(100, 2, 100),
            'High': np.random.normal(102, 2, 100),
            'Low': np.random.normal(98, 2, 100),
            'Close': np.random.normal(100, 2, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        })
        
        # Sample options data with high put/call imbalance
        self.bearish_options_data = pd.DataFrame({
            'type': ['put', 'put', 'call', 'put'],
            'volume': [2000, 1500, 500, 1000],
            'premium': [300000, 225000, 75000, 150000]
        })
        
        # Sample options data with balanced flow
        self.neutral_options_data = pd.DataFrame({
            'type': ['call', 'put', 'call', 'put'],
            'volume': [1000, 1000, 1000, 1000],
            'premium': [150000, 150000, 150000, 150000]
        })
        
        # Sample dark pool data with large blocks
        self.dark_pool_data = pd.DataFrame({
            'price': [100.0, 101.0, 99.0],
            'volume': [500000, 300000, 200000]
        })
        
        # Sample greeks data indicating high risk
        self.high_risk_greeks: Dict[str, float] = {
            'gamma': 0.05,
            'vega': 0.4,
            'theta': -0.2,
            'delta': 0.9
        }
        
        # Sample greeks data indicating normal risk
        self.normal_greeks: Dict[str, float] = {
            'gamma': 0.002,
            'vega': 0.15,
            'theta': -0.05,
            'delta': 0.6
        }
        
        # Create trending price data
        self.trending_data = self.price_data.copy()
        self.trending_data['Close'] = 100 + np.linspace(0, 10, 100)
        
        # Create ranging price data
        self.ranging_data = self.price_data.copy()
        self.ranging_data['Close'] = 100 + np.sin(np.linspace(0, 4*np.pi, 100))
        
        # Create volatile price data
        self.volatile_data = self.price_data.copy()
        self.volatile_data['Close'] = 100 + np.random.normal(0, 5, 100)
        
        # Sample market tide data
        self.market_tide = {
            'sentiment': 'bullish',
            'score': 0.75
        }
        
        # Sample options data
        self.options_data = pd.DataFrame({
            'type': ['call', 'put', 'call', 'put'],
            'volume': [1500, 500, 1200, 800],
            'premium': [150000, 50000, 120000, 80000]
        })
        
    def test_analyze_trending_market(self):
        result = self.analyzer.analyze(
            self.trending_data,
            options_data=self.neutral_options_data,
            dark_pool_data=self.dark_pool_data,
            greeks_data=self.normal_greeks
        )
        
        self.assertEqual(result.regime, 'trending')
        self.assertGreater(result.trend_strength, 0)
        self.assertTrue(0 <= result.confidence <= 1)
        self.assertIn('options_flow', result.metrics)
        self.assertIn('dark_pool', result.metrics)
        self.assertIn('greeks', result.metrics)
        
    def test_analyze_ranging_market(self):
        result = self.analyzer.analyze(
            self.ranging_data,
            options_data=self.neutral_options_data,
            dark_pool_data=self.dark_pool_data,
            greeks_data=self.normal_greeks
        )
        
        self.assertEqual(result.regime, 'ranging')
        self.assertLess(abs(result.trend_strength), self.analyzer.trend_threshold)
        self.assertGreater(result.confidence, 0.5)  # Higher confidence due to balanced options flow
        
    def test_analyze_volatile_market(self):
        result = self.analyzer.analyze(
            self.volatile_data,
            options_data=self.bearish_options_data,
            dark_pool_data=self.dark_pool_data,
            greeks_data=self.high_risk_greeks
        )
        
        self.assertEqual(result.volatility_regime, 'high')
        self.assertTrue(len(result.support_resistance) <= 3)
        self.assertLess(result.confidence, 0.5)  # Lower confidence due to high risk metrics
        
    def test_market_tide_integration(self):
        result = self.analyzer.analyze(
            self.trending_data,
            market_tide=self.market_tide,
            options_data=self.neutral_options_data,
            dark_pool_data=self.dark_pool_data,
            greeks_data=self.normal_greeks
        )
        
        self.assertEqual(result.regime, 'trending')
        self.assertGreater(result.confidence, 0.6)
        
    def test_risk_metrics_integration(self):
        # Test with high risk conditions
        high_risk_result = self.analyzer.analyze(
            self.volatile_data,
            options_data=self.bearish_options_data,
            dark_pool_data=self.dark_pool_data,
            greeks_data=self.high_risk_greeks
        )
        
        self.assertLess(high_risk_result.confidence, 0.5)
        self.assertIn('high_risk_factors', high_risk_result.metrics)
        self.assertGreater(len(high_risk_result.metrics['high_risk_factors']), 0)
        
        # Test with normal risk conditions
        normal_risk_result = self.analyzer.analyze(
            self.trending_data,
            options_data=self.neutral_options_data,
            dark_pool_data=self.dark_pool_data,
            greeks_data=self.normal_greeks
        )
        
        self.assertGreater(normal_risk_result.confidence, 0.5)
        self.assertEqual(len(normal_risk_result.metrics.get('high_risk_factors', [])), 0)
        
    def test_support_resistance_levels(self):
        result = self.analyzer.analyze(self.ranging_data)
        
        self.assertIsInstance(result.support_resistance, list)
        self.assertLessEqual(len(result.support_resistance), 3)
        if result.support_resistance:
            self.assertTrue(all(isinstance(x, float) for x in result.support_resistance))
            
    def test_empty_data(self):
        empty_data = pd.DataFrame()
        result = self.analyzer.analyze(empty_data)
        
        self.assertEqual(result.regime, 'unknown')
        self.assertEqual(result.trend_strength, 0.0)
        self.assertEqual(result.confidence, 0.0)
        
    def test_momentum_calculation(self):
        result = self.analyzer.analyze(self.trending_data)
        
        self.assertGreater(result.momentum_score, 0)
        self.assertTrue(-1 <= result.momentum_score <= 1)

if __name__ == '__main__':
    unittest.main()
