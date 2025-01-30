import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class PositionConfig:
    max_position_size: float = 0.1
    max_risk_per_trade: float = 0.02
    atr_multiplier: float = 2.0
    trailing_stop_activation: float = 0.02
    min_profit_target: float = 0.015

class RiskManager:
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.config = PositionConfig()
        
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr[-period:])
        return float(atr)
        
    def calculate_position_size(self, current_price: float, volatility: float, 
                              market_regime: str) -> Tuple[float, Dict[str, float]]:
        vol_factor = 1.0 - (volatility / 100)
        regime_factors = {'trending': 1.0, 'ranging': 0.7, 'high_volatility': 0.5}
        regime_factor = regime_factors.get(market_regime, 0.5)
        
        base_position = self.portfolio_value * self.config.max_position_size
        adjusted_position = base_position * vol_factor * regime_factor
        
        risk_amount = self.portfolio_value * self.config.max_risk_per_trade
        max_shares = risk_amount / (current_price * volatility)
        
        final_position = min(adjusted_position, max_shares * current_price)
        shares = int(final_position / current_price)
        
        metrics = {
            'position_value': shares * current_price,
            'position_size_pct': (shares * current_price) / self.portfolio_value,
            'risk_amount': risk_amount,
            'volatility_factor': vol_factor,
            'regime_factor': regime_factor
        }
        
        return shares, metrics
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, 
                          atr: float) -> Tuple[float, float]:
        atr_stop = atr * self.config.atr_multiplier
        
        if position_type == 'long':
            stop_loss = entry_price - atr_stop
            profit_target = entry_price + (atr_stop * 1.5)
        else:
            stop_loss = entry_price + atr_stop
            profit_target = entry_price - (atr_stop * 1.5)
            
        return stop_loss, profit_target
    
    def update_trailing_stop(self, position_type: str, current_price: float, 
                           entry_price: float, highest_price: float, 
                           lowest_price: float, current_stop: float) -> Optional[float]:
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.config.trailing_stop_activation:
                new_stop = highest_price * (1 - self.config.trailing_stop_activation)
                return max(new_stop, current_stop) if current_stop else new_stop
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= self.config.trailing_stop_activation:
                new_stop = lowest_price * (1 + self.config.trailing_stop_activation)
                return min(new_stop, current_stop) if current_stop else new_stop
        
        return current_stop
    
    def validate_trade(self, entry_price: float, stop_loss: float, 
                      profit_target: float, position_type: str) -> Tuple[bool, str]:
        if position_type == 'long':
            risk_pct = (entry_price - stop_loss) / entry_price
            reward_pct = (profit_target - entry_price) / entry_price
        else:
            risk_pct = (stop_loss - entry_price) / entry_price
            reward_pct = (entry_price - profit_target) / entry_price
            
        risk_reward_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        if risk_pct > self.config.max_risk_per_trade:
            return False, "Risk exceeds maximum allowed"
        if risk_reward_ratio < 1.5:
            return False, "Risk:Reward ratio below minimum threshold"
        if reward_pct < self.config.min_profit_target:
            return False, "Profit target too small"
            
        return True, "Trade validated"
