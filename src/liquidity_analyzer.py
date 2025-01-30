import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class LiquidityMetrics:
    spread: float
    depth: float
    resilience: float
    volume_profile: float
    composite_score: float

class LiquidityAnalyzer:
    def __init__(self, min_volume_threshold: float = 1000):
        self.min_volume_threshold = min_volume_threshold
        
    def analyze_liquidity(self, 
                         price_data: pd.DataFrame,
                         volume_data: pd.DataFrame,
                         target_position_size: float) -> Dict[str, float]:
        metrics = self._calculate_liquidity_metrics(
            price_data, volume_data, target_position_size
        )
        
        timing_score = self._calculate_timing_score(metrics)
        execution_windows = self._identify_execution_windows(
            volume_data, metrics, target_position_size
        )
        
        return {
            'metrics': {
                'spread': metrics.spread,
                'depth': metrics.depth,
                'resilience': metrics.resilience,
                'volume_profile': metrics.volume_profile,
                'composite_score': metrics.composite_score
            },
            'timing_score': timing_score,
            'execution_windows': execution_windows,
            'recommendation': self._generate_recommendation(
                timing_score, execution_windows
            )
        }
        
    def _calculate_liquidity_metrics(self,
                                   price_data: pd.DataFrame,
                                   volume_data: pd.DataFrame,
                                   target_position_size: float) -> LiquidityMetrics:
        # Calculate bid-ask spread
        spread = np.mean(price_data['high'] - price_data['low']) / price_data['close'].mean()
        
        # Calculate market depth
        avg_volume = volume_data['volume'].mean()
        depth = min(1.0, avg_volume / (target_position_size * 10))
        
        # Calculate market resilience
        price_impact = np.abs(price_data['close'].pct_change())
        volume_ratio = volume_data['volume'].pct_change()
        resilience = 1.0 - np.corrcoef(price_impact, volume_ratio)[0, 1]
        
        # Calculate volume profile
        volume_profile = volume_data['volume'].rolling(window=20).mean() / avg_volume
        volume_score = float(volume_profile.mean())
        
        # Calculate composite score
        composite_score = (
            0.3 * (1.0 - spread) +  # Lower spread is better
            0.3 * depth +           # Higher depth is better
            0.2 * resilience +      # Higher resilience is better
            0.2 * volume_score      # Higher volume is better
        )
        
        return LiquidityMetrics(
            spread=spread,
            depth=depth,
            resilience=resilience,
            volume_profile=volume_score,
            composite_score=composite_score
        )
        
    def _calculate_timing_score(self, metrics: LiquidityMetrics) -> float:
        base_score = metrics.composite_score
        
        # Adjust score based on specific metrics
        if metrics.spread > 0.01:  # High spread penalty
            base_score *= 0.8
        if metrics.depth < 0.3:    # Low depth penalty
            base_score *= 0.7
        if metrics.resilience < 0.5:  # Low resilience penalty
            base_score *= 0.9
            
        return float(np.clip(base_score, 0.0, 1.0))
        
    def _identify_execution_windows(self,
                                  volume_data: pd.DataFrame,
                                  metrics: LiquidityMetrics,
                                  target_size: float) -> List[Dict]:
        volume_profile = volume_data['volume'].rolling(window=20).mean()
        high_liquidity_periods = volume_profile > self.min_volume_threshold
        
        windows = []
        current_window = None
        
        for timestamp, is_liquid in high_liquidity_periods.items():
            if is_liquid and current_window is None:
                current_window = {'start': timestamp, 'score': 0.0}
            elif not is_liquid and current_window is not None:
                current_window['end'] = timestamp
                current_window['score'] = self._calculate_window_score(
                    volume_data.loc[current_window['start']:current_window['end']],
                    metrics,
                    target_size
                )
                windows.append(current_window)
                current_window = None
                
        return sorted(windows, key=lambda x: x['score'], reverse=True)
        
    def _calculate_window_score(self,
                              window_data: pd.DataFrame,
                              metrics: LiquidityMetrics,
                              target_size: float) -> float:
        avg_volume = window_data['volume'].mean()
        volume_coverage = min(1.0, avg_volume / target_size)
        
        return float(
            0.5 * metrics.composite_score +
            0.5 * volume_coverage
        )
        
    def _generate_recommendation(self,
                               timing_score: float,
                               execution_windows: List[Dict]) -> Dict:
        if not execution_windows:
            return {
                'action': 'delay',
                'confidence': 0.0,
                'reason': 'No suitable execution windows found'
            }
            
        best_window = execution_windows[0]
        confidence = timing_score * best_window['score']
        
        if confidence > 0.8:
            action = 'execute'
            reason = 'High liquidity conditions'
        elif confidence > 0.5:
            action = 'partial_execute'
            reason = 'Moderate liquidity conditions'
        else:
            action = 'delay'
            reason = 'Poor liquidity conditions'
            
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'best_window': best_window
        }
