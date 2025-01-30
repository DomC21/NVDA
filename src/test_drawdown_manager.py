import unittest
from drawdown_manager import DrawdownManager

class TestDrawdownManager(unittest.TestCase):
    def setUp(self):
        self.initial_value = 100000.0
        self.drawdown_manager = DrawdownManager(self.initial_value)
        
    def test_calculate_drawdown(self):
        current_value = 95000.0  # 5% drawdown
        drawdown, max_drawdown = self.drawdown_manager.calculate_drawdown(current_value)
        
        self.assertAlmostEqual(drawdown, 0.05)
        self.assertAlmostEqual(max_drawdown, 0.05)
        
    def test_get_position_adjustment(self):
        # No drawdown
        adjustment = self.drawdown_manager.get_position_adjustment(0.0)
        self.assertEqual(adjustment, 1.0)
        
        # Approaching max drawdown
        adjustment = self.drawdown_manager.get_position_adjustment(0.07)
        self.assertLess(adjustment, 1.0)
        self.assertGreater(adjustment, 0.0)
        
        # Max drawdown exceeded
        adjustment = self.drawdown_manager.get_position_adjustment(0.10)
        self.assertEqual(adjustment, 0.0)
        
    def test_validate_trade(self):
        # Normal conditions
        is_valid, message, metrics = self.drawdown_manager.validate_trade(95000.0)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Trade validated")
        
        # Approaching max drawdown
        is_valid, message, metrics = self.drawdown_manager.validate_trade(89000.0)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Approaching maximum drawdown limit")
        
        # Max drawdown exceeded
        is_valid, message, metrics = self.drawdown_manager.validate_trade(85000.0)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Maximum drawdown exceeded")
        
    def test_check_recovery(self):
        # Initial state
        has_recovered, recovery_pct = self.drawdown_manager.check_recovery(self.initial_value)
        self.assertTrue(has_recovered)
        
        # Create drawdown
        self.drawdown_manager.calculate_drawdown(90000.0)
        
        # Not enough recovery
        has_recovered, recovery_pct = self.drawdown_manager.check_recovery(91000.0)
        self.assertFalse(has_recovered)
        
        # Sufficient recovery
        has_recovered, recovery_pct = self.drawdown_manager.check_recovery(95000.0)
        self.assertTrue(has_recovered)

if __name__ == '__main__':
    unittest.main()
