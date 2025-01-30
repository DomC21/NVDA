import unittest
import numpy as np
import pandas as pd
from model_validation import TimeSeriesValidator
from tensorflow.keras.models import Sequential

class TestTimeSeriesValidator(unittest.TestCase):
    def setUp(self):
        self.validator = TimeSeriesValidator(n_splits=3)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        self.data = pd.DataFrame({
            'close': np.random.randn(500).cumsum(),
            'volume': np.random.randint(1000, 10000, 500),
            'rsi': np.random.uniform(0, 100, 500),
            'macd': np.random.randn(500),
            'bb_upper': np.random.randn(500).cumsum() + 2,
            'bb_lower': np.random.randn(500).cumsum() - 2
        })
        
    def test_prepare_sequences(self):
        X, y = self.validator.prepare_sequences(self.data, sequence_length=60)
        
        self.assertEqual(X.shape[0], len(self.data) - 60)
        self.assertEqual(X.shape[1], 60)
        self.assertEqual(X.shape[2], self.data.shape[1])
        self.assertEqual(y.shape[0], len(self.data) - 60)
        
    def test_create_lstm_model(self):
        X, _ = self.validator.prepare_sequences(self.data)
        model = self.validator.create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 10)
        
    def test_create_gru_model(self):
        X, _ = self.validator.prepare_sequences(self.data)
        model = self.validator.create_gru_model(input_shape=(X.shape[1], X.shape[2]))
        
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 10)
        
    def test_cross_validation(self):
        X, y = self.validator.prepare_sequences(self.data)
        results = self.validator.cross_validate(X, y)
        
        self.assertEqual(len(results['fold_metrics']), 3)
        self.assertEqual(len(results['ensemble_metrics']), 3)
        self.assertEqual(len(results['predictions']), 3)
        
        for metrics in results['fold_metrics']:
            self.assertIn('lstm', metrics)
            self.assertIn('gru', metrics)
            self.assertIn('xgboost', metrics)
            self.assertIn('gradient_boosting', metrics)
            
            for model_metrics in metrics.values():
                self.assertIn('mae', model_metrics)
                self.assertIn('mape', model_metrics)
                self.assertIn('rmse', model_metrics)
        
    def test_ensemble_predict(self):
        X, y = self.validator.prepare_sequences(self.data)
        
        # Train models on a small subset
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train models
        lstm_model = self.validator.create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        lstm_model.compile(optimizer='adam', loss='huber')
        gru_model = self.validator.create_gru_model(input_shape=(X.shape[1], X.shape[2]))
        gru_model.compile(optimizer='adam', loss='huber')
        
        self.validator.train_neural_network(lstm_model, X_train, y_train, X_test, y_test)
        self.validator.train_neural_network(gru_model, X_train, y_train, X_test, y_test)
        
        xgb_model = self.validator.train_xgboost(X_train, y_train, X_test, y_test)
        gb_model = self.validator.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        models = {
            'lstm': lstm_model,
            'gru': gru_model,
            'xgboost': xgb_model,
            'gradient_boosting': gb_model
        }
        
        ensemble_pred, individual_preds = self.validator.ensemble_predict(models, X_test)
        
        self.assertEqual(len(ensemble_pred), len(X_test))
        self.assertEqual(len(individual_preds), 4)
        for pred in individual_preds.values():
            self.assertEqual(len(pred), len(X_test))

if __name__ == '__main__':
    unittest.main()
