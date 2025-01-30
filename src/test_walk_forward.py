import unittest
import pandas as pd
import numpy as np
from walk_forward import WalkForwardOptimizer

class TestWalkForwardOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = WalkForwardOptimizer(
            train_window=100,
            test_window=20,
            step_size=10,
            min_train_size=50
        )
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, len(dates)),
            'feature2': np.random.normal(0, 1, len(dates)),
            'target': np.random.normal(0, 1, len(dates))
        }, index=dates)
        
    def test_window_generation(self):
        windows = self.optimizer._generate_windows(self.data)
        
        self.assertTrue(len(windows) > 0)
        for train_idx, test_idx in windows:
            self.assertTrue(len(train_idx) >= self.optimizer.min_train_size)
            self.assertTrue(len(test_idx) <= self.optimizer.test_window)
            self.assertEqual(train_idx[-1] + 1, test_idx[0])
            
    def test_optimization(self):
        def dummy_model(**params):
            return type('DummyModel', (), {
                'param1': params.get('param1', 0),
                'param2': params.get('param2', 0)
            })
            
        def dummy_scoring(model, data):
            return -abs(model.param1 - 0.5) - abs(model.param2 - 0.5)
            
        param_grid = {
            'param1': [0.1, 0.5, 0.9],
            'param2': [0.1, 0.5, 0.9]
        }
        
        result = self.optimizer.optimize(
            self.data,
            dummy_model,
            param_grid,
            dummy_scoring
        )
        
        self.assertEqual(len(result.train_scores), len(result.test_scores))
        self.assertEqual(len(result.optimization_params), len(result.train_scores))
        self.assertTrue(0 <= result.robustness_score <= 1)
        
    def test_param_combinations(self):
        param_grid = {
            'a': [1, 2],
            'b': [3, 4]
        }
        
        combinations = self.optimizer._generate_param_combinations(param_grid)
        
        self.assertEqual(len(combinations), 4)
        self.assertTrue({'a': 1, 'b': 3} in combinations)
        self.assertTrue({'a': 1, 'b': 4} in combinations)
        self.assertTrue({'a': 2, 'b': 3} in combinations)
        self.assertTrue({'a': 2, 'b': 4} in combinations)
        
    def test_edge_cases(self):
        # Test with small dataset
        small_data = self.data.iloc[:10]
        windows = self.optimizer._generate_windows(small_data)
        self.assertEqual(len(windows), 0)
        
        # Test with invalid scoring function
        def failing_scoring(model, data):
            raise Exception("Scoring failed")
            
        score = self.optimizer._evaluate_model(None, self.data, failing_scoring)
        self.assertEqual(score, float('-inf'))

if __name__ == '__main__':
    unittest.main()
