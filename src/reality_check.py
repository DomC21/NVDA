import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SignalValidation:
    is_valid: bool
    confidence: float
    failure_reasons: List[str]
    market_context: Dict[str, float]

class RealityCheck:
    def __init__(self,
                 min_volume_threshold: float = 100000,
                 max_spread_threshold: float = 0.03,
                 min_confidence: float = 0.6):
        self.min_volume_threshold = min_volume_threshold
        self.max_spread_threshold = max_spread_threshold
        self.min_confidence = min_confidence
        
    def validate_signal(self,
                       signal: Dict,
                       market_data: pd.DataFrame,
                       options_data: Optional[pd.DataFrame] = None) -> SignalValidation:
        failures = []
        context = {}
        
        volume_check = self._check_volume(market_data)
        if not volume_check['valid']:
            failures.append(volume_check['reason'])
            
        spread_check = self._check_spread(market_data)
        if not spread_check['valid']:
            failures.append(spread_check['reason'])
            
        trend_check = self._check_trend_alignment(signal, market_data)
        if not trend_check['valid']:
            failures.append(trend_check['reason'])
            
        if options_data is not None:
            flow_check = self._check_options_flow(signal, options_data)
            if not flow_check['valid']:
                failures.append(flow_check['reason'])
                context.update(flow_check['context'])
                
        volatility_check = self._check_volatility(market_data)
        if not volatility_check['valid']:
            failures.append(volatility_check['reason'])
            
        context.update({
            'volume_ratio': volume_check['context']['volume_ratio'],
            'spread': spread_check['context']['spread'],
            'trend_strength': trend_check['context']['strength'],
            'volatility': volatility_check['context']['volatility']
        })
        
        confidence = self._calculate_confidence(failures, context)
        
        return SignalValidation(
            is_valid=len(failures) == 0 and confidence >= self.min_confidence,
            confidence=confidence,
            failure_reasons=failures,
            market_context=context
        )
        
    def _check_volume(self, market_data: pd.DataFrame) -> Dict:
        recent_volume = market_data['Volume'].iloc[-1]
        avg_volume = market_data['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = recent_volume / avg_volume
        
        return {
            'valid': recent_volume >= self.min_volume_threshold,
            'reason': f'Volume {recent_volume:.0f} below threshold {self.min_volume_threshold:.0f}',
            'context': {'volume_ratio': volume_ratio}
        }
        
    def _check_spread(self, market_data: pd.DataFrame) -> Dict:
        if 'Ask' in market_data and 'Bid' in market_data:
            spread = (market_data['Ask'].iloc[-1] - market_data['Bid'].iloc[-1]) / \
                    market_data['Bid'].iloc[-1]
        else:
            spread = (market_data['High'].iloc[-1] - market_data['Low'].iloc[-1]) / \
                    market_data['Low'].iloc[-1]
            
        return {
            'valid': spread <= self.max_spread_threshold,
            'reason': f'Spread {spread:.4f} above threshold {self.max_spread_threshold:.4f}',
            'context': {'spread': spread}
        }
        
    def _check_trend_alignment(self,
                             signal: Dict,
                             market_data: pd.DataFrame) -> Dict:
        returns = market_data['Close'].pct_change()
        sma_20 = market_data['Close'].rolling(window=20).mean()
        sma_50 = market_data['Close'].rolling(window=50).mean()
        
        trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        if signal['direction'] == 'buy':
            aligned = trend_strength > 0
        else:
            aligned = trend_strength < 0
            
        return {
            'valid': aligned,
            'reason': f'Signal {signal["direction"]} misaligned with trend',
            'context': {'strength': abs(trend_strength)}
        }
        
    def _check_options_flow(self,
                          signal: Dict,
                          options_data: pd.DataFrame) -> Dict:
        recent_flow = options_data.iloc[-10:]
        
        call_volume = recent_flow['call_volume'].sum()
        put_volume = recent_flow['put_volume'].sum()
        call_put_ratio = call_volume / put_volume if put_volume > 0 else float('inf')
        
        if signal['direction'] == 'buy':
            aligned = call_put_ratio > 1.5
        else:
            aligned = call_put_ratio < 0.67
            
        return {
            'valid': aligned,
            'reason': f'Options flow ({call_put_ratio:.2f}) misaligned with signal',
            'context': {
                'call_put_ratio': call_put_ratio,
                'total_volume': call_volume + put_volume
            }
        }
        
    def _check_volatility(self, market_data: pd.DataFrame) -> Dict:
        returns = market_data['Close'].pct_change()
        current_vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
        avg_vol = returns.rolling(window=60).std().iloc[-1] * np.sqrt(252)
        
        vol_ratio = current_vol / avg_vol
        
        return {
            'valid': vol_ratio < 2.0,
            'reason': f'Volatility ({vol_ratio:.2f}x) too high relative to average',
            'context': {'volatility': current_vol}
        }
        
    def _calculate_confidence(self,
                            failures: List[str],
                            context: Dict[str, float]) -> float:
        if failures:
            return max(0.0, 1.0 - 0.2 * len(failures))
            
        confidence = 1.0
        
        # Volume contribution
        volume_score = min(1.0, context['volume_ratio'] / 2.0)
        confidence *= 0.3 * volume_score
        
        # Spread contribution
        spread_score = 1.0 - (context['spread'] / self.max_spread_threshold)
        confidence *= 0.2 * max(0.0, spread_score)
        
        # Trend contribution
        trend_score = min(1.0, context['trend_strength'] * 5)
        confidence *= 0.3 * trend_score
        
        # Volatility contribution
        vol_score = 1.0 - min(1.0, context['volatility'] / 0.5)
        confidence *= 0.2 * vol_score
        
        return float(confidence)
