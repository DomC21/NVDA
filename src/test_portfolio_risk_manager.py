import unittest
from portfolio_risk_manager import PortfolioRiskManager

class TestPortfolioRiskManager(unittest.TestCase):
    def setUp(self):
        self.initial_value = 1000000.0
        self.portfolio_manager = PortfolioRiskManager(self.initial_value)
        
    def test_add_remove_position(self):
        self.portfolio_manager.add_position(
            symbol='NVDA',
            value=100000.0,
            sector='Technology',
            beta=1.5,
            correlation_data={'AAPL': 0.6, 'AMD': 0.8}
        )
        
        self.assertIn('NVDA', self.portfolio_manager.positions)
        self.assertEqual(self.portfolio_manager.positions['NVDA']['value'], 100000.0)
        
        self.portfolio_manager.remove_position('NVDA')
        self.assertNotIn('NVDA', self.portfolio_manager.positions)
        
    def test_calculate_portfolio_risk(self):
        self.portfolio_manager.add_position(
            symbol='NVDA',
            value=100000.0,
            sector='Technology',
            beta=1.5,
            correlation_data={'AAPL': 0.6}
        )
        
        self.portfolio_manager.add_position(
            symbol='AAPL',
            value=100000.0,
            sector='Technology',
            beta=1.2,
            correlation_data={'NVDA': 0.6}
        )
        
        risk, metrics = self.portfolio_manager.calculate_portfolio_risk()
        
        self.assertGreater(risk, 0.0)
        self.assertIn('diversification', metrics)
        self.assertIn('concentration', metrics)
        self.assertIn('correlation', metrics)
        
    def test_validate_new_position(self):
        # Add initial position
        self.portfolio_manager.add_position(
            symbol='NVDA',
            value=200000.0,
            sector='Technology',
            beta=1.5,
            correlation_data={}
        )
        
        # Test sector exposure limit
        is_valid, message, metrics = self.portfolio_manager.validate_new_position(
            symbol='AAPL',
            value=300000.0,
            sector='Technology',
            beta=1.2,
            correlation_data={'NVDA': 0.6}
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "Sector exposure limit exceeded")
        
        # Test leverage limit
        is_valid, message, metrics = self.portfolio_manager.validate_new_position(
            symbol='JPM',
            value=1000000.0,
            sector='Finance',
            beta=1.1,
            correlation_data={'NVDA': 0.3}
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "Leverage limit exceeded")
        
    def test_position_size_adjustment(self):
        adjustment = self.portfolio_manager.get_position_size_adjustment(0.05)
        self.assertEqual(adjustment, 1.0)
        
        adjustment = self.portfolio_manager.get_position_size_adjustment(0.15)
        self.assertEqual(adjustment, 0.0)
        
        adjustment = self.portfolio_manager.get_position_size_adjustment(0.12)
        self.assertGreater(adjustment, 0.0)
        self.assertLess(adjustment, 1.0)

if __name__ == '__main__':
    unittest.main()
