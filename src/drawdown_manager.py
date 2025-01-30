import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class DrawdownConfig:
    max_drawdown_pct: float = 0.10  # Maximum allowed drawdown (10%)
    recovery_threshold_pct: float = 0.05  # Required recovery before re-entry (5%)
    lookback_period: int = 20  # Days to look back for drawdown calculation
    risk_reduction_factor: float = 0.5  # Reduce position size when approaching max drawdown

class DrawdownManager:
    def __init__(self, initial_portfolio_value: float):
        self.initial_value = initial_portfolio_value
        self.peak_value = initial_portfolio_value
        self.config = DrawdownConfig()
        self.drawdown_history: List[float] = []
        
    def calculate_drawdown(self, current_value: float) -> Tuple[float, float]:
        """Calculate current drawdown and update peak value."""
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.drawdown_history.append(drawdown)
        
        if len(self.drawdown_history) > self.config.lookback_period:
            self.drawdown_history.pop(0)
            
        max_drawdown = max(self.drawdown_history)
        return drawdown, max_drawdown
        
    def get_position_adjustment(self, current_drawdown: float) -> float:
        """Calculate position size adjustment based on drawdown level."""
        if current_drawdown >= self.config.max_drawdown_pct:
            return 0.0  # Stop trading
        elif current_drawdown >= self.config.max_drawdown_pct * 0.7:
            # Reduce position size as we approach max drawdown
            reduction = (current_drawdown / self.config.max_drawdown_pct) * self.config.risk_reduction_factor
            return max(0.0, 1.0 - reduction)
        return 1.0
        
    def validate_trade(self, current_value: float) -> Tuple[bool, str, Dict[str, float]]:
        """Validate trade based on drawdown metrics."""
        current_drawdown, max_drawdown = self.calculate_drawdown(current_value)
        position_adjustment = self.get_position_adjustment(current_drawdown)
        
        metrics = {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'position_adjustment': position_adjustment,
            'peak_value': self.peak_value
        }
        
        if current_drawdown >= self.config.max_drawdown_pct:
            return False, "Maximum drawdown exceeded", metrics
            
        if max_drawdown >= self.config.max_drawdown_pct * 0.8:
            return False, "Approaching maximum drawdown limit", metrics
            
        return True, "Trade validated", metrics
        
    def check_recovery(self, current_value: float) -> Tuple[bool, float]:
        """Check if portfolio has recovered enough to resume trading."""
        if not self.drawdown_history:
            return True, 1.0
            
        recent_low = self.peak_value * (1 - max(self.drawdown_history))
        recovery_pct = (current_value - recent_low) / recent_low
        
        return recovery_pct >= self.config.recovery_threshold_pct, recovery_pct
