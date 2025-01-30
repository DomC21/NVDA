import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class WalkForwardResult:
    train_scores: List[float]
    test_scores: List[float]
    optimization_params: List[Dict]
    robustness_score: float

class WalkForwardOptimizer:
    def __init__(self,
                 train_window: int = 252,  # 1 year
                 test_window: int = 63,    # 3 months
                 step_size: int = 21,      # 1 month
                 min_train_size: int = 126):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        
    def optimize(self,
                data: pd.DataFrame,
                model_fn: Callable,
                param_grid: Dict,
                scoring_fn: Callable) -> WalkForwardResult:
        windows = self._generate_windows(data)
        results = []
        
        for train_idx, test_idx in windows:
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            window_result = self._optimize_window(
                train_data, test_data, model_fn, param_grid, scoring_fn
            )
            results.append(window_result)
            
        return self._aggregate_results(results)
        
    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        windows = []
        data_size = len(data)
        
        for start in range(0, data_size - self.test_window, self.step_size):
            if start + self.train_window > data_size:
                break
                
            train_end = start + self.train_window
            test_end = min(train_end + self.test_window, data_size)
            
            if train_end - start < self.min_train_size:
                continue
                
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(train_end, test_end)
            windows.append((train_idx, test_idx))
            
        return windows
        
    def _optimize_window(self,
                        train_data: pd.DataFrame,
                        test_data: pd.DataFrame,
                        model_fn: Callable,
                        param_grid: Dict,
                        scoring_fn: Callable) -> Dict:
        best_score = float('-inf')
        best_params = None
        best_test_score = None
        
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            model = model_fn(**params)
            train_score = self._evaluate_model(model, train_data, scoring_fn)
            
            if train_score > best_score:
                best_score = train_score
                best_params = params
                best_test_score = self._evaluate_model(model, test_data, scoring_fn)
                
        return {
            'train_score': best_score,
            'test_score': best_test_score,
            'params': best_params
        }
        
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []
        
        def recurse(params: Dict, depth: int):
            if depth == len(keys):
                combinations.append(params.copy())
                return
                
            for value in values[depth]:
                params[keys[depth]] = value
                recurse(params, depth + 1)
                
        recurse({}, 0)
        return combinations
        
    def _evaluate_model(self,
                       model: object,
                       data: pd.DataFrame,
                       scoring_fn: Callable) -> float:
        try:
            return scoring_fn(model, data)
        except Exception as e:
            return float('-inf')
            
    def _aggregate_results(self, results: List[Dict]) -> WalkForwardResult:
        train_scores = [r['train_score'] for r in results]
        test_scores = [r['test_score'] for r in results]
        params = [r['params'] for r in results]
        
        # Calculate robustness score
        score_diffs = np.array(train_scores) - np.array(test_scores)
        robustness = 1.0 / (1.0 + np.std(score_diffs))
        
        return WalkForwardResult(
            train_scores=train_scores,
            test_scores=test_scores,
            optimization_params=params,
            robustness_score=float(robustness)
        )
