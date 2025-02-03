import unittest
import numpy as np
from typing import Dict, Any, Optional
from risk_manager import RiskManager, PositionConfig

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.portfolio_value = 100000.0
        self.risk_manager = RiskManager(self.portfolio_value)
        self.sample_greeks: Dict[str, float] = {
            'gamma': 0.002,
            'vega': 0.15,
            'theta': -0.05,
            'delta': 0.6,
            'hedge_ratio': 0.8
        }
        self.high_risk_greeks: Dict[str, float] = {
            'gamma': 0.05,
            'vega': 0.4,
            'theta': -0.2,
            'delta': 0.9,
            'hedge_ratio': 0.2
        }
        self.config = PositionConfig()
        
    def test_calculate_position_size(self):
        current_price = 500.0
        volatility = 20.0
        market_regime = 'trending'
        
        # Test without Greeks data
        shares, metrics = self.risk_manager.calculate_position_size(
            current_price, volatility, market_regime
        )
        
        self.assertIsInstance(shares, int)
        self.assertGreater(shares, 0)
        self.assertLess(metrics['position_size_pct'], 0.1)
        self.assertGreater(metrics['risk_amount'], 0)
        
        # Test with normal Greeks data
        shares, metrics = self.risk_manager.calculate_position_size(
            current_price, volatility, market_regime, self.sample_greeks
        )
        
        self.assertIsInstance(shares, int)
        self.assertGreater(shares, 0)
        self.assertIn('greek_adjustment', metrics)
        self.assertGreater(metrics['greek_adjustment']['adjustment_factor'], 0.5)
        
        # Test with high risk Greeks data
        shares, metrics = self.risk_manager.calculate_position_size(
            current_price, volatility, market_regime, self.high_risk_greeks
        )
        
        self.assertIsInstance(shares, int)
        self.assertGreater(shares, 0)
        self.assertLess(metrics['greek_adjustment']['adjustment_factor'], 0.5)
        self.assertLess(metrics['position_size_pct'], 0.1)
        self.assertGreater(metrics['risk_amount'], 0)
        
    def test_calculate_stop_loss(self):
        entry_price = 500.0
        atr = 10.0
        
        stop_loss, profit_target = self.risk_manager.calculate_stop_loss(
            entry_price, 'long', atr
        )
        
        self.assertLess(stop_loss, entry_price)
        self.assertGreater(profit_target, entry_price)
        
        stop_loss, profit_target = self.risk_manager.calculate_stop_loss(
            entry_price, 'short', atr
        )
        
        self.assertGreater(stop_loss, entry_price)
        self.assertLess(profit_target, entry_price)
        
    def test_validate_trade(self):
        entry_price = 500.0
        stop_loss = 490.0
        profit_target = 525.0
        
        # Test without Greeks data
        is_valid, message = self.risk_manager.validate_trade(
            entry_price, stop_loss, profit_target, 'long'
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Trade validated")
        
        # Test with normal Greeks data
        is_valid, message = self.risk_manager.validate_trade(
            entry_price, stop_loss, profit_target, 'long', self.sample_greeks
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Trade validated")
        
        # Test with high risk Greeks data
        is_valid, message = self.risk_manager.validate_trade(
            entry_price, stop_loss, profit_target, 'long', self.high_risk_greeks
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Excessive Greeks exposure", message)
        
        # Test invalid trade with poor risk-reward
        is_valid, message = self.risk_manager.validate_trade(
            entry_price, 485.0, 505.0, 'long'
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "Risk:Reward ratio below minimum threshold")
        
    def test_adjust_for_greeks(self):
        position_size = 50000.0
        
        # Test with normal Greeks data
        adjusted_size, metrics = self.risk_manager.adjust_for_greeks(
            position_size, self.sample_greeks
        )
        
        self.assertGreater(adjusted_size, 0)
        self.assertLess(adjusted_size, position_size)
        self.assertGreater(metrics['adjustment_factor'], 0.5)
        self.assertIn('exposures', metrics)
        self.assertIn('reasons', metrics)
        
        # Test with high risk Greeks data
        adjusted_size, metrics = self.risk_manager.adjust_for_greeks(
            position_size, self.high_risk_greeks
        )
        
        self.assertGreater(adjusted_size, 0)
        self.assertLess(adjusted_size, position_size)
        self.assertLess(metrics['adjustment_factor'], 0.5)
        self.assertGreater(len(metrics['reasons']), 0)
        
        # Test with empty Greeks data
        adjusted_size, metrics = self.risk_manager.adjust_for_greeks(
            position_size, {}
        )
        
        self.assertEqual(adjusted_size, position_size)
        self.assertEqual(metrics['adjustment_factor'], 1.0)
        
    def test_update_trailing_stop(self):
        entry_price = 500.0
        current_price = 520.0
        highest_price = 525.0
        lowest_price = 495.0
        current_stop = 490.0
        
        new_stop = self.risk_manager.update_trailing_stop(
            'long', current_price, entry_price,
            highest_price, lowest_price, current_stop
        )
        
        self.assertIsNotNone(new_stop)
        if new_stop is not None:
            self.assertGreater(new_stop, current_stop)

if __name__ == '__main__':
    unittest.main()
