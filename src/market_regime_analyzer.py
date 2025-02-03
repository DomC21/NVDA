import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class MarketRegimeAnalysis:
    regime: str  # 'trending', 'ranging', 'volatile'
    trend_strength: float
    volatility_regime: str
    momentum_score: float
    support_resistance: List[float]
    confidence: float

class MarketRegimeAnalyzer:
    def __init__(self, 
                lookback_period: int = 20,
                volatility_window: int = 20,
                trend_threshold: float = 0.02):
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        
    def analyze(
            self,
            price_data: pd.DataFrame,
            market_tide: Optional[Dict] = None,
            options_data: Optional[pd.DataFrame] = None) -> MarketRegimeAnalysis:
        if price_data.empty:
            return self._create_empty_analysis()
            
        trend_strength = self._calculate_trend_strength(price_data)
        volatility_regime = self._analyze_volatility(price_data)
        momentum = self._calculate_momentum(price_data)
        support_resistance = self._find_support_resistance(price_data)
        
        regime = self._determine_regime(
            trend_strength, volatility_regime, momentum,
            market_tide, options_data
        )
        
        confidence = self._calculate_confidence(
            trend_strength, volatility_regime, momentum,
            market_tide, options_data
        )
        
        return MarketRegimeAnalysis(
            regime=regime,
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            momentum_score=momentum,
            support_resistance=support_resistance,
            confidence=confidence
        )
        
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        if len(data) < self.lookback_period:
            return 0.0
            
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        
        trend_direction = np.sign(sma_20.iloc[-1] - sma_50.iloc[-1])
        price_change = ((data['Close'].iloc[-1] - 
                        data['Close'].iloc[-self.lookback_period]) / 
                       data['Close'].iloc[-self.lookback_period])
        
        return trend_direction * abs(price_change)
        
    def _analyze_volatility(self, data: pd.DataFrame) -> str:
        if len(data) < self.volatility_window:
            return 'normal'
            
        returns = data['Close'].pct_change()
        current_vol = returns.rolling(window=self.volatility_window).std().iloc[-1]
        historical_vol = returns.std()
        
        if current_vol > historical_vol * 1.5:
            return 'high'
        elif current_vol < historical_vol * 0.5:
            return 'low'
        return 'normal'
        
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        if len(data) < self.lookback_period:
            return 0.0
            
        momentum_periods = [5, 10, 20]
        momentum_scores = []
        
        for period in momentum_periods:
            if len(data) >= period:
                returns = float(
                    (data['Close'].iloc[-1] / data['Close'].iloc[-period]) - 1)
                momentum_scores.append(returns)
                
        return float(np.mean(momentum_scores)) if momentum_scores else 0.0
        
    def _find_support_resistance(self, data: pd.DataFrame) -> List[float]:
        if len(data) < 20:
            return []
            
        highs = data['High'].rolling(window=20, center=True).max()
        lows = data['Low'].rolling(window=20, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(20, len(data) - 20):
            if highs.iloc[i] == data['High'].iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
            if lows.iloc[i] == data['Low'].iloc[i]:
                support_levels.append(data['Low'].iloc[i])
                
        levels = sorted(set(support_levels + resistance_levels))
        return levels[-3:] if levels else []
        
    def _determine_regime(self,
                         trend_strength: float,
                         volatility_regime: str,
                         momentum: float,
                         market_tide: Optional[Dict],
                         options_data: Optional[pd.DataFrame]) -> str:
        # Start with price-based regime
        if abs(trend_strength) > self.trend_threshold:
            base_regime = 'trending'
        elif volatility_regime == 'high':
            base_regime = 'volatile'
        else:
            base_regime = 'ranging'
            
        # Adjust based on market tide
        if market_tide and 'score' in market_tide:
            if abs(market_tide['score'] - 0.5) > 0.2:
                if base_regime == 'ranging':
                    base_regime = 'trending'
                    
        # Consider options flow
        if options_data is not None and not options_data.empty:
            call_volume = (options_data[options_data['type'] == 'call']
                         ['volume'].sum())
            put_volume = (options_data[options_data['type'] == 'put']
                         ['volume'].sum())
            if (abs(call_volume - put_volume) / 
                (call_volume + put_volume) > 0.3):
                if base_regime == 'ranging':
                    base_regime = 'trending'
                    
        return base_regime
        
    def _calculate_confidence(self,
                            trend_strength: float,
                            volatility_regime: str,
                            momentum: float,
                            market_tide: Optional[Dict],
                            options_data: Optional[pd.DataFrame]) -> float:
        confidence_factors: List[float] = []
        
        # Trend strength confidence
        confidence_factors.append(min(1.0, abs(trend_strength) / self.trend_threshold))
        
        # Volatility regime confidence
        vol_confidence = {
            'high': 0.8,
            'normal': 0.6,
            'low': 0.4
        }
        confidence_factors.append(vol_confidence[volatility_regime])
        
        # Momentum confidence
        confidence_factors.append(min(1.0, abs(momentum) * 5))
        
        # Market tide confidence
        if market_tide and 'score' in market_tide:
            confidence_factors.append(abs(float(market_tide['score']) - 0.5) * 2)
            
        # Options flow confidence
        if options_data is not None and not options_data.empty:
            call_volume = float(options_data[options_data['type'] == 'call']
                                       ['volume'].sum())
            put_volume = float(options_data[options_data['type'] == 'put']
                             ['volume'].sum())
            total_volume = call_volume + put_volume
            if total_volume > 0:
                options_confidence = abs(call_volume - put_volume) / total_volume
                confidence_factors.append(options_confidence)
                
        return float(np.mean(confidence_factors))
        
    def _create_empty_analysis(self) -> MarketRegimeAnalysis:
        return MarketRegimeAnalysis(
            regime='unknown',
            trend_strength=0.0,
            volatility_regime='normal',
            momentum_score=0.0,
            support_resistance=[],
            confidence=0.0
        )
