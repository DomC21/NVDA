import pandas as pd
from typing import Dict
from datetime import datetime, time

class TradingTimeOptimizer:
    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self.optimal_periods = {
            'entry': {'time': None, 'score': 0},
            'exit': {'time': None, 'score': 0}
        }
        
    def analyze_timing(self, data: pd.DataFrame) -> Dict[str, Dict]:
        hourly_returns = self._calculate_hourly_returns(data)
        volume_profile = self._analyze_volume_profile(data)
        volatility_profile = self._analyze_volatility_profile(data)
        
        entry_scores = self._calculate_entry_scores(
            hourly_returns, volume_profile, volatility_profile)
        exit_scores = self._calculate_exit_scores(
            hourly_returns, volume_profile, volatility_profile)
            
        self.optimal_periods = {
            'entry': self._get_optimal_time(entry_scores),
            'exit': self._get_optimal_time(exit_scores)
        }
        
        return self.optimal_periods
        
    def _calculate_hourly_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['hour'] = pd.to_datetime(data.index).hour
        hourly_returns = data.groupby('hour')['Close'].pct_change()
        return pd.DataFrame({'returns': hourly_returns})
        
    def _analyze_volume_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['hour'] = pd.to_datetime(data.index).hour
        volume_profile = data.groupby('hour')['Volume'].mean()
        normalized = volume_profile / volume_profile.max()
        return pd.DataFrame({'volume': normalized})
        
    def _analyze_volatility_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['hour'] = pd.to_datetime(data.index).hour
        returns = data.groupby('hour')['Close'].pct_change()
        volatility = returns.groupby(data['hour']).std()
        normalized = volatility / volatility.max()
        return pd.DataFrame({'volatility': normalized})
        
    def _calculate_entry_scores(
            self,
            returns: pd.DataFrame,
            volume: pd.DataFrame,
            volatility: pd.DataFrame
    ) -> pd.DataFrame:
        scores = pd.DataFrame()
        scores['return_score'] = (returns + 1).rank(pct=True)
        scores['volume_score'] = volume.rank(pct=True)
        scores['volatility_score'] = (1 - volatility).rank(pct=True)
        
        weights = {'return_score': 0.4, 'volume_score': 0.4, 'volatility_score': 0.2}
        scores['total_score'] = sum(
            scores[col] * weight for col, weight in weights.items())
        return scores
        
    def _calculate_exit_scores(
            self,
            returns: pd.DataFrame,
            volume: pd.DataFrame,
            volatility: pd.DataFrame
    ) -> pd.DataFrame:
        scores = pd.DataFrame()
        scores['return_score'] = (-returns + 1).rank(pct=True)
        scores['volume_score'] = volume.rank(pct=True)
        scores['volatility_score'] = volatility.rank(pct=True)
        
        weights = {'return_score': 0.4, 'volume_score': 0.4, 'volatility_score': 0.2}
        scores['total_score'] = sum(
            scores[col] * weight for col, weight in weights.items())
        return scores
        
    def _get_optimal_time(self, scores: pd.DataFrame) -> Dict:
        best_hour = scores['total_score'].idxmax()
        return {
            'time': time(hour=int(best_hour)),
            'score': scores.loc[best_hour, 'total_score']
        }
        
    def get_timing_recommendation(self, current_time: datetime,
                                side: str) -> Dict[str, float]:
        if not self.optimal_periods[side]['time']:
            return {'execute_now': True, 'confidence': 0.5}
            
        optimal_hour = self.optimal_periods[side]['time'].hour
        current_hour = current_time.hour
        
        hour_diff = abs(current_hour - optimal_hour)
        if hour_diff > 12:
            hour_diff = 24 - hour_diff
            
        confidence = max(0, 1 - (hour_diff / 4))
        execute_now = hour_diff <= 1
        
        return {
            'execute_now': execute_now,
            'confidence': confidence,
            'optimal_hour': optimal_hour,
            'current_hour': current_hour
        }
