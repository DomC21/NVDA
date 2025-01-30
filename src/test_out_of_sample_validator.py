import unittest
import pandas as pd
import numpy as np
from out_of_sample_validator import OutOfSampleValidator

class TestOutOfSampleValidator(unittest.TestCase):
    def setUp(self):
        self.validator = OutOfSampleValidator()
        
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        self.test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, len(dates)),
            'feature2': np.random.normal(0, 1, len(dates)),
            'target': np.random.normal(0, 1, len(dates))
        }, index=dates)
        
    def test_temporal_splits(self):
        train, val, test = self.validator._create_temporal_splits(self.test_data)
        
        expected_train_size = int(len(self.test_data) * self.validator.train_ratio)
        expected_val_size = int(len(self.test_data) * self.validator.validation_ratio)
        
        self.assertEqual(len(train), expected_train_size)
        self.assertEqual(len(val), expected_val_size)
        self.assertTrue(train.index[0] < val.index[0] < test.index[0])
        
    def test_validation(self):
        def dummy_model(**params):
            return type('DummyModel', (), {
                'predict': lambda x: np.zeros(len(x))
            })
            
        def dummy_scoring(model, data):
            return 0.5
            
        metrics = self.validator.validate(
            self.test_data,
            dummy_model,
            dummy_scoring,
            {}
        )
        
        self.assertTrue(hasattr(metrics, 'in_sample_score'))
        self.assertTrue(hasattr(metrics, 'out_sample_score'))
        self.assertTrue(hasattr(metrics, 'generalization_ratio'))
        self.assertTrue(hasattr(metrics, 'stability_score'))
        self.assertTrue(hasattr(metrics, 'prediction_bias'))
        
    def test_cross_validation(self):
        def dummy_model(**params):
            return type('DummyModel', (), {
                'predict': lambda x: np.zeros(len(x))
            })
            
        def dummy_scoring(model, data):
            return 0.5
            
        results = self.validator.cross_validate(
            self.test_data,
            dummy_model,
            dummy_scoring,
            {},
            n_splits=5
        )
        
        self.assertEqual(len(results['in_sample_scores']), 5)
        self.assertEqual(len(results['out_sample_scores']), 5)
        self.assertEqual(len(results['generalization_ratios']), 5)
        self.assertEqual(len(results['stability_scores']), 5)
        self.assertEqual(len(results['prediction_biases']), 5)
        
    def test_edge_cases(self):
        def failing_model(**params):
            return type('FailingModel', (), {
                'predict': lambda x: (_ for _ in ()).throw(Exception('Model failed'))
            })
            
        def failing_scoring(model, data):
            raise Exception('Scoring failed')
            
        metrics = self.validator.validate(
            self.test_data,
            failing_model,
            failing_scoring,
            {}
        )
        
        self.assertEqual(metrics.in_sample_score, float('-inf'))
        self.assertEqual(metrics.out_sample_score, float('-inf'))
        self.assertEqual(metrics.generalization_ratio, 0)
        self.assertEqual(metrics.stability_score, 0)
        self.assertEqual(metrics.prediction_bias, 0)

if __name__ == '__main__':
    unittest.main()
