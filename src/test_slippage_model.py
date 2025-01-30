import unittest
from slippage_model import SlippageModel

class TestSlippageModel(unittest.TestCase):
    def setUp(self):
        self.model = SlippageModel()
        self.market_data = {
            'price': 100.0,
            'avg_volume': 1000000,
            'volatility': 0.02,
            'side': 'buy'
        }
        
    def test_estimate_transaction_cost(self):
        costs = self.model.estimate_transaction_cost(
            price=100.0,
            size=1000,
            market_data=self.market_data
        )
        
        self.assertIn('spread_cost', costs)
        self.assertIn('market_impact', costs)
        self.assertIn('volatility_cost', costs)
        self.assertIn('commission', costs)
        self.assertIn('slippage_cost', costs)
        self.assertIn('total_cost', costs)
        
        self.assertTrue(costs['total_cost'] > 0)
        self.assertTrue(costs['slippage_cost'] >= 0)
        
    def test_optimize_order_size(self):
        target_size = 1000
        max_slippage = 0.003
        
        optimal_size = self.model.optimize_order_size(
            target_size=target_size,
            max_slippage=max_slippage,
            market_data=self.market_data
        )
        
        self.assertTrue(optimal_size > 0)
        self.assertTrue(optimal_size <= target_size)
        
        costs = self.model.estimate_transaction_cost(
            price=self.market_data['price'],
            size=optimal_size,
            market_data=self.market_data
        )
        
        slippage_ratio = costs['slippage_cost'] / (self.market_data['price'] * optimal_size)
        self.assertTrue(slippage_ratio <= max_slippage)
        
    def test_edge_cases(self):
        market_data = {
            'price': 100.0,
            'avg_volume': 0,
            'volatility': 0,
            'side': 'buy'
        }
        
        costs = self.model.estimate_transaction_cost(
            price=100.0,
            size=1,
            market_data=market_data
        )
        
        self.assertTrue(costs['total_cost'] > 0)
        self.assertTrue(costs['commission'] >= 1.0)

if __name__ == '__main__':
    unittest.main()
