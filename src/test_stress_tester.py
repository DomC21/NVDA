import unittest
import pandas as pd
import numpy as np
from stress_tester import StressTester, StressScenario

class TestStressTester(unittest.TestCase):
    def setUp(self):
        self.tester = StressTester()
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'Close': 100 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod(),
            'Volume': np.random.normal(1000000, 100000, len(dates)),
            'Bid-Ask': np.random.uniform(0.01, 0.03, len(dates))
        }, index=dates)
        
        self.position_sizes = np.ones(len(dates))
        self.risk_limits = {
            'max_drawdown': 0.15,
            'var_limit': 0.10
        }
        
    def test_scenario_generation(self):
        scenarios = self.tester._generate_base_scenarios()
        
        self.assertTrue(len(scenarios) > 0)
        for scenario in scenarios:
            self.assertIsInstance(scenario, StressScenario)
            self.assertTrue(hasattr(scenario, 'name'))
            self.assertTrue(hasattr(scenario, 'price_shock'))
            
    def test_stress_application(self):
        scenario = StressScenario(
            name="Test Scenario",
            price_shock=-0.10,
            volatility_multiplier=2.0,
            volume_multiplier=1.5,
            correlation_change=0.2,
            liquidity_shock=-0.3
        )
        
        stressed_data = self.tester._apply_stress_scenario(
            self.test_data, scenario)
            
        self.assertLess(
            stressed_data['Close'].mean(),
            self.test_data['Close'].mean()
        )
        self.assertGreater(
            stressed_data['Volume'].mean(),
            self.test_data['Volume'].mean()
        )
        
    def test_strategy_evaluation(self):
        scenario = self.tester.scenarios[0]
        stressed_data = self.tester._apply_stress_scenario(
            self.test_data, scenario)
            
        result = self.tester._evaluate_strategy(
            stressed_data,
            self.position_sizes,
            self.risk_limits,
            scenario
        )
        
        self.assertEqual(result.scenario_name, scenario.name)
        self.assertTrue(result.max_drawdown <= 0)
        self.assertTrue(result.recovery_time >= 0)
        self.assertTrue(-1 <= result.sharpe_ratio <= 10)
        self.assertTrue(0 <= result.liquidity_score <= 1)
        self.assertTrue(0 <= result.survival_score <= 1)
        
    def test_full_stress_test(self):
        results = self.tester.run_stress_tests(
            self.test_data,
            self.position_sizes,
            self.risk_limits
        )
        
        self.assertEqual(len(results), len(self.tester.scenarios))
        for result in results:
            self.assertTrue(hasattr(result, 'scenario_name'))
            self.assertTrue(hasattr(result, 'max_drawdown'))
            self.assertTrue(hasattr(result, 'recovery_time'))
            
    def test_edge_cases(self):
        # Test with constant price data
        constant_data = self.test_data.copy()
        constant_data['Close'] = 100
        
        results = self.tester.run_stress_tests(
            constant_data,
            self.position_sizes,
            self.risk_limits
        )
        
        for result in results:
            self.assertEqual(result.sharpe_ratio, 0)

if __name__ == '__main__':
    unittest.main()
