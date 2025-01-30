import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class RegimeMetrics:
    regime: MarketRegime
    confidence: float
    volatility: float
    trend_strength: float
    momentum_score: float

class MarketRegimeDetector:
    def __init__(self, price_data: pd.DataFrame, 
                 lookback_period: int = 20,
                 volatility_threshold: float = 0.015,
                 trend_threshold: float = 0.6):
        self.data = price_data
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
    def detect_regime(self) -> RegimeMetrics:
        if len(self.data) < self.lookback_period:
            return RegimeMetrics(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                volatility=0.0,
                trend_strength=0.0,
                momentum_score=0.0
            )
            
        returns = self.data['Close'].pct_change()
        volatility = returns.rolling(window=self.lookback_period).std().iloc[-1]
        
        price_changes = self.data['Close'].diff()
        positive_moves = (price_changes > 0).rolling(window=self.lookback_period).sum().iloc[-1]
        trend_strength = abs(positive_moves / self.lookback_period - 0.5) * 2
        
        rsi = self._calculate_rsi()
        macd, signal = self._calculate_macd()
        momentum_score = self._calculate_momentum_score(rsi, macd, signal)
        
        if volatility > self.volatility_threshold * 2:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(1.0, volatility / (self.volatility_threshold * 3))
        elif volatility < self.volatility_threshold * 0.5:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = min(1.0, (self.volatility_threshold - volatility) / self.volatility_threshold)
        elif trend_strength > self.trend_threshold:
            if momentum_score > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            confidence = min(1.0, trend_strength / self.trend_threshold)
        else:
            regime = MarketRegime.RANGING
            confidence = min(1.0, (self.trend_threshold - trend_strength) / self.trend_threshold)
            
        return RegimeMetrics(
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength,
            momentum_score=momentum_score
        )
        
    def _calculate_rsi(self, period: int = 14) -> float:
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))
        
    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        exp1 = self.data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]
        
    def _calculate_momentum_score(self, rsi: float, macd: float, signal: float) -> float:
        rsi_score = (rsi - 50) / 50
        macd_score = 1 if macd > signal else -1
        return (rsi_score + macd_score) / 2
        
    def get_regime_parameters(self) -> Dict[str, Dict[str, float]]:
        regime = self.detect_regime()
        
        parameters = {
            'position_sizing': {
                'trending_up': 1.0,
                'trending_down': 0.8,
                'ranging': 0.6,
                'high_volatility': 0.4,
                'low_volatility': 0.8
            },
            'stop_loss': {
                'trending_up': 2.0,
                'trending_down': 1.5,
                'ranging': 1.0,
                'high_volatility': 3.0,
                'low_volatility': 1.0
            },
            'take_profit': {
                'trending_up': 3.0,
                'trending_down': 2.0,
                'ranging': 1.5,
                'high_volatility': 4.0,
                'low_volatility': 1.5
            }
        }
        
        return {
            'current_regime': regime.regime.value,
            'confidence': regime.confidence,
            'position_size_factor': parameters['position_sizing'][regime.regime.value],
            'stop_loss_factor': parameters['stop_loss'][regime.regime.value],
            'take_profit_factor': parameters['take_profit'][regime.regime.value]
        }
