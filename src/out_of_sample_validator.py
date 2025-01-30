import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ValidationMetrics:
    in_sample_score: float
    out_sample_score: float
    generalization_ratio: float
    stability_score: float
    prediction_bias: float

class OutOfSampleValidator:
    def __init__(self,
                 train_ratio: float = 0.6,
                 validation_ratio: float = 0.2,
                 test_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        
    def validate(self,
                data: pd.DataFrame,
                model_fn: Callable,
                scoring_fn: Callable,
                hyperparameters: Dict) -> ValidationMetrics:
        splits = self._create_temporal_splits(data)
        train_data, val_data, test_data = splits
        
        model = model_fn(**hyperparameters)
        
        in_sample_score = self._evaluate_model(
            model, train_data, val_data, scoring_fn)
            
        out_sample_score = scoring_fn(model, test_data)
        
        stability = self._assess_stability(
            model, [train_data, val_data, test_data], scoring_fn)
            
        bias = self._calculate_prediction_bias(
            model, test_data)
            
        generalization = out_sample_score / in_sample_score \
                        if in_sample_score != 0 else 0
                        
        return ValidationMetrics(
            in_sample_score=in_sample_score,
            out_sample_score=out_sample_score,
            generalization_ratio=generalization,
            stability_score=stability,
            prediction_bias=bias
        )
        
    def _create_temporal_splits(self,
                              data: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         pd.DataFrame]:
        total_size = len(data)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.validation_ratio)
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]
        
        return train_data, val_data, test_data
        
    def _evaluate_model(self,
                       model: object,
                       train_data: pd.DataFrame,
                       val_data: pd.DataFrame,
                       scoring_fn: Callable) -> float:
        try:
            train_score = scoring_fn(model, train_data)
            val_score = scoring_fn(model, val_data)
            return (train_score + val_score) / 2
        except Exception as e:
            return float('-inf')
            
    def _assess_stability(self,
                         model: object,
                         data_splits: List[pd.DataFrame],
                         scoring_fn: Callable) -> float:
        scores = []
        for split in data_splits:
            try:
                score = scoring_fn(model, split)
                scores.append(score)
            except Exception:
                scores.append(float('-inf'))
                
        if not scores or all(s == float('-inf') for s in scores):
            return 0.0
            
        score_std = np.std([s for s in scores if s != float('-inf')])
        score_mean = np.mean([s for s in scores if s != float('-inf')])
        
        cv = score_std / score_mean if score_mean != 0 else float('inf')
        stability = 1.0 / (1.0 + cv)
        
        return float(stability)
        
    def _calculate_prediction_bias(self,
                                 model: object,
                                 test_data: pd.DataFrame) -> float:
        try:
            predictions = model.predict(test_data)
            actuals = test_data['target'].values
            bias = np.mean(predictions - actuals)
            return float(bias)
        except Exception:
            return 0.0
            
    def cross_validate(self,
                      data: pd.DataFrame,
                      model_fn: Callable,
                      scoring_fn: Callable,
                      hyperparameters: Dict,
                      n_splits: int = 5) -> Dict[str, List[float]]:
        results = {
            'in_sample_scores': [],
            'out_sample_scores': [],
            'generalization_ratios': [],
            'stability_scores': [],
            'prediction_biases': []
        }
        
        split_size = len(data) // n_splits
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            
            test_data = data.iloc[start_idx:end_idx]
            train_data = pd.concat([
                data.iloc[:start_idx],
                data.iloc[end_idx:]
            ])
            
            metrics = self.validate(
                train_data,
                model_fn,
                scoring_fn,
                hyperparameters
            )
            
            results['in_sample_scores'].append(metrics.in_sample_score)
            results['out_sample_scores'].append(metrics.out_sample_score)
            results['generalization_ratios'].append(metrics.generalization_ratio)
            results['stability_scores'].append(metrics.stability_score)
            results['prediction_biases'].append(metrics.prediction_bias)
            
        return results
