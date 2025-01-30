import unittest
import numpy as np
from monte_carlo import MonteCarloSimulator

class TestMonteCarloSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = MonteCarloSimulator(
            n_simulations=100,
            time_horizon=252,
            confidence_level=0.95
        )
        
        # Generate sample historical returns
        self.historical_returns = np.random.normal(0.0001, 0.02, 1000)
        
    def test_simulate_returns(self):
        result = self.simulator.simulate_returns(
            self.historical_returns,
            initial_value=100000.0
        )
        
        self.assertEqual(len(result.returns), self.simulator.n_simulations)
        self.assertEqual(result.drawdowns.shape, 
                        (self.simulator.n_simulations, self.simulator.time_horizon))
        self.assertTrue(isinstance(result.sharpe_ratio, float))
        self.assertTrue(isinstance(result.var_95, float))
        self.assertTrue(isinstance(result.cvar_95, float))
        
    def test_assess_strategy_risk(self):
        position_sizes = np.ones(self.simulator.time_horizon)
        stop_losses = np.full(self.simulator.time_horizon, 0.02)
        
        result = self.simulator.assess_strategy_risk(
            self.historical_returns,
            position_sizes,
            stop_losses
        )
        
        self.assertIn('expected_return', result)
        self.assertIn('expected_drawdown', result)
        self.assertIn('var_95', result)
        self.assertIn('cvar_95', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('success_rate', result)
        
        self.assertTrue(isinstance(result['expected_return'], float))
        self.assertTrue(isinstance(result['expected_drawdown'], float))
        self.assertTrue(-1 <= result['expected_drawdown'] <= 0)
        self.assertTrue(0 <= result['success_rate'] <= 1)
        
    def test_risk_controls(self):
        returns = np.array([0.01, -0.05, 0.02, 0.01, -0.03])
        position_sizes = np.ones_like(returns)
        stop_losses = np.full_like(returns, 0.04)
        
        adjusted_returns = self.simulator._apply_risk_controls(
            returns, position_sizes, stop_losses
        )
        
        self.assertTrue(np.all(adjusted_returns[2:] == 0))
        
    def test_edge_cases(self):
        # Test with zero returns
        zero_returns = np.zeros(1000)
        result = self.simulator.simulate_returns(zero_returns)
        self.assertTrue(np.allclose(result.returns.mean(), 0, atol=1e-3))
        
        # Test with very high volatility
        high_vol_returns = np.random.normal(0, 0.5, 1000)
        result = self.simulator.simulate_returns(high_vol_returns)
        self.assertTrue(result.var_95 < 0)

if __name__ == '__main__':
    unittest.main()
