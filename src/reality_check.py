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
                 min_confidence: float = 0.6,
                 max_gamma_exposure: float = 0.01,
                 max_vega_exposure: float = 0.02,
                 max_theta_decay: float = 0.005,
                 min_delta_hedge: float = 0.3,
                 high_block_threshold: float = 1_000_000):
        self.min_volume_threshold = min_volume_threshold
        self.max_spread_threshold = max_spread_threshold
        self.min_confidence = min_confidence
        self.max_gamma_exposure = max_gamma_exposure
        self.max_vega_exposure = max_vega_exposure
        self.max_theta_decay = max_theta_decay
        self.min_delta_hedge = min_delta_hedge
        self.high_block_threshold = high_block_threshold
        
    def validate_signal(self,
                       signal: Dict,
                       market_data: pd.DataFrame,
                       options_data: Optional[pd.DataFrame] = None,
                       dark_pool_data: Optional[pd.DataFrame] = None,
                       greeks_data: Optional[Dict[str, float]] = None) -> SignalValidation:
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
                
        if dark_pool_data is not None:
            dark_pool_check = self._check_dark_pool_alignment(signal, dark_pool_data)
            if not dark_pool_check['valid']:
                failures.append(dark_pool_check['reason'])
                context.update(dark_pool_check['context'])
                
        if greeks_data is not None:
            greeks_check = self._check_greeks_exposure(greeks_data)
            if not greeks_check['valid']:
                failures.append(greeks_check['reason'])
                context.update(greeks_check['context'])
                
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
        
    def _check_dark_pool_alignment(self,
                                 signal: Dict,
                                 dark_pool_data: pd.DataFrame) -> Dict:
        if dark_pool_data.empty:
            return {'valid': True, 'reason': '', 'context': {}}
            
        major_blocks = dark_pool_data.nlargest(3, 'volume')
        block_sum = major_blocks['volume'].sum()
        avg_block_price = (major_blocks['price'] * major_blocks['volume']).sum() / block_sum
        
        context = {
            'block_sum': block_sum,
            'avg_block_price': avg_block_price
        }
        
        if block_sum > self.high_block_threshold:
            if signal['direction'] == 'buy' and avg_block_price < dark_pool_data['price'].iloc[-1]:
                return {
                    'valid': False,
                    'reason': f'Large dark pool blocks ({block_sum:,.0f}) below current price',
                    'context': context
                }
            elif signal['direction'] == 'sell' and avg_block_price > dark_pool_data['price'].iloc[-1]:
                return {
                    'valid': False,
                    'reason': f'Large dark pool blocks ({block_sum:,.0f}) above current price',
                    'context': context
                }
                
        return {'valid': True, 'reason': '', 'context': context}
        
    def _check_greeks_exposure(self, greeks: Dict[str, float]) -> Dict:
        reasons = []
        context = {}
        
        gamma = abs(greeks.get('gamma', 0))
        if gamma > self.max_gamma_exposure:
            reasons.append(f'High gamma exposure ({gamma:.4f})')
            
        vega = abs(greeks.get('vega', 0))
        if vega > self.max_vega_exposure:
            reasons.append(f'High vega exposure ({vega:.4f})')
            
        theta = abs(greeks.get('theta', 0))
        if theta > self.max_theta_decay:
            reasons.append(f'High theta decay ({theta:.4f})')
            
        hedge_ratio = greeks.get('hedge_ratio', 0)
        if hedge_ratio < self.min_delta_hedge:
            reasons.append(f'Low delta hedge ratio ({hedge_ratio:.2f})')
            
        context.update({
            'gamma_exposure': gamma,
            'vega_exposure': vega,
            'theta_decay': theta,
            'hedge_ratio': hedge_ratio
        })
        
        return {
            'valid': len(reasons) == 0,
            'reason': '; '.join(reasons) if reasons else '',
            'context': context
        }
        
    def _calculate_confidence(self,
                            failures: List[str],
                            context: Dict[str, float]) -> float:
        if failures:
            return max(0.0, 1.0 - 0.2 * len(failures))
            
        confidence = 1.0
        
        # Volume contribution (20%)
        volume_score = min(1.0, context.get('volume_ratio', 1.0) / 2.0)
        confidence *= 0.2 * volume_score
        
        # Spread contribution (15%)
        spread_score = 1.0 - (context.get('spread', 0) / self.max_spread_threshold)
        confidence *= 0.15 * max(0.0, spread_score)
        
        # Trend contribution (20%)
        trend_score = min(1.0, context.get('trend_strength', 0) * 5)
        confidence *= 0.2 * trend_score
        
        # Volatility contribution (15%)
        vol_score = 1.0 - min(1.0, context.get('volatility', 0) / 0.5)
        confidence *= 0.15 * vol_score
        
        # Greeks contribution (15%)
        if 'gamma_exposure' in context:
            greeks_score = 1.0
            greeks_score *= 1.0 - min(1.0, context['gamma_exposure'] / self.max_gamma_exposure)
            greeks_score *= 1.0 - min(1.0, context['vega_exposure'] / self.max_vega_exposure)
            greeks_score *= 1.0 - min(1.0, context['theta_decay'] / self.max_theta_decay)
            confidence *= 0.15 * greeks_score
            
        # Dark pool contribution (15%)
        if 'block_sum' in context:
            block_score = 1.0 - min(1.0, context['block_sum'] / self.high_block_threshold)
            confidence *= 0.15 * block_score
        
        return float(confidence)
