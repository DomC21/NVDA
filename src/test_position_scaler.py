import unittest
from position_scaler import PositionScaler

class TestPositionScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = PositionScaler(
            base_position_size=1000,
            max_position_size=2000,
            min_position_size=100
        )
        
    def test_calculate_position_size(self):
        result = self.scaler.calculate_position_size(
            signal_confidence=80.0,
            volatility=0.02,
            avg_volume=1000000,
            current_price=100.0,
            transaction_costs={
                'market_impact': 50.0,
                'total_cost': 100.0
            }
        )
        
        self.assertIn('initial_entry', result)
        self.assertIn('reserve', result)
        self.assertIn('total_size', result)
        self.assertIn('scaling_factors', result)
        
        self.assertTrue(result['initial_entry'] > 0)
        self.assertTrue(result['reserve'] > 0)
        self.assertEqual(
            result['total_size'],
            result['initial_entry'] + result['reserve']
        )
        
        self.assertTrue(result['total_size'] <= self.scaler.max_position_size)
        self.assertTrue(result['total_size'] >= self.scaler.min_position_size)
        
    def test_scaling_factors(self):
        result = self.scaler.calculate_position_size(
            signal_confidence=90.0,
            volatility=0.01,
            avg_volume=2000000,
            current_price=100.0,
            transaction_costs={
                'market_impact': 25.0,
                'total_cost': 50.0
            }
        )
        
        factors = result['scaling_factors']
        self.assertIn('confidence', factors)
        self.assertIn('volatility', factors)
        self.assertIn('liquidity', factors)
        self.assertIn('market_impact', factors)
        self.assertIn('cost_efficiency', factors)
        
        for factor in factors.values():
            self.assertTrue(0 <= factor <= 1)
            
    def test_edge_cases(self):
        result = self.scaler.calculate_position_size(
            signal_confidence=0.0,
            volatility=1.0,
            avg_volume=0,
            current_price=100.0,
            transaction_costs={
                'market_impact': 1000.0,
                'total_cost': 2000.0
            }
        )
        
        self.assertEqual(result['total_size'], self.scaler.min_position_size)

if __name__ == '__main__':
    unittest.main()
