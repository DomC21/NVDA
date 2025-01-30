import unittest
import numpy as np
from risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.portfolio_value = 100000.0
        self.risk_manager = RiskManager(self.portfolio_value)
        
    def test_calculate_position_size(self):
        current_price = 500.0
        volatility = 20.0
        market_regime = 'trending'
        
        shares, metrics = self.risk_manager.calculate_position_size(
            current_price, volatility, market_regime
        )
        
        self.assertGreater(shares, 0)
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
        
        is_valid, message = self.risk_manager.validate_trade(
            entry_price, stop_loss, profit_target, 'long'
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Trade validated")
        
        # Test invalid trade with poor risk-reward
        is_valid, message = self.risk_manager.validate_trade(
            entry_price, 485.0, 505.0, 'long'
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "Risk:Reward ratio below minimum threshold")
        
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
        self.assertGreaterEqual(new_stop, current_stop)

if __name__ == '__main__':
    unittest.main()
