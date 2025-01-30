import unittest
from src.price_predictor import PricePredictor
import numpy as np
import pandas as pd

class TestPricePredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = PricePredictor()
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.predictor)
        self.assertEqual(self.predictor.sequence_length, 60)
        
    def test_feature_extraction(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100)
        sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        features = self.predictor._extract_features(sample_data)
        self.assertIsNotNone(features)
        self.assertIn('Close', features.columns)
        self.assertIn('RSI', features.columns)
        self.assertIn('MACD', features.columns)
        
    def test_price_ranges(self):
        current_price = 100.0
        predicted_price = 105.0
        ranges = self.predictor.generate_price_ranges(current_price, predicted_price)
        
        self.assertEqual(ranges['prediction'], predicted_price)
        self.assertIn('90%', ranges['confidence_ranges'])
        self.assertIn('70%', ranges['confidence_ranges'])
        self.assertIn('50%', ranges['confidence_ranges'])
        
        # Check range bounds
        lower_90, upper_90 = ranges['confidence_ranges']['90%']
        self.assertLess(lower_90, predicted_price)
        self.assertGreater(upper_90, predicted_price)
        
    def test_backtest_results(self):
        results = self.predictor.backtest(test_size=0.2)
        
        # Check required metrics exist
        self.assertIn('mae', results)
        self.assertIn('mape', results)
        self.assertIn('rmse', results)
        self.assertIn('predictions', results)
        self.assertIn('actuals', results)
        
        # Validate metric values
        self.assertGreater(results['mae'], 0)
        self.assertGreater(results['mape'], 0)
        self.assertGreater(results['rmse'], 0)
        
        # Check predictions array
        self.assertEqual(len(results['predictions']), len(results['actuals']))
        
    def test_model_prediction(self):
        # Create sample sequence data
        sequence_length = self.predictor.sequence_length
        sample_data = pd.DataFrame({
            'Close': np.random.randn(sequence_length + 10).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, sequence_length + 10)
        })
        
        features = self.predictor._extract_features(sample_data)
        self.predictor.build_model(input_shape=(sequence_length, features.shape[1]))
        
        prediction = self.predictor.predict_next_day(features)
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, (float, np.float32, np.float64))
        self.assertGreater(prediction, 0)

if __name__ == '__main__':
    unittest.main()
