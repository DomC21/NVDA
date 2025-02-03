import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

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
        self.max_gamma_exposure = 0.01  # Maximum allowed gamma exposure per $1 of portfolio value
        self.max_vega_exposure = 0.02   # Maximum allowed vega exposure per $1 of portfolio value
        self.max_theta_decay = 0.005    # Maximum allowed theta decay per day as % of portfolio
        self.min_delta_hedge = 0.3      # Minimum delta hedge ratio for option positions
        
    def adjust_for_greeks(self, position_size: float, greeks_data: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """Adjusts position size based on option Greeks exposure."""
        if not greeks_data:
            return position_size, {'adjustment_factor': 1.0, 'reason': 'No Greeks data available'}
            
        # Calculate total exposures
        gamma_exposure = abs(greeks_data.get('gamma', 0) * position_size)
        vega_exposure = abs(greeks_data.get('vega', 0) * position_size)
        theta_exposure = abs(greeks_data.get('theta', 0) * position_size)
        delta_exposure = abs(greeks_data.get('delta', 0) * position_size)
        
        # Initialize adjustment metrics
        adjustment_factor = 1.0
        reasons = []
        
        # Check gamma exposure
        gamma_limit = self.portfolio_value * self.max_gamma_exposure
        if gamma_exposure > gamma_limit:
            gamma_adj = gamma_limit / gamma_exposure
            adjustment_factor = min(adjustment_factor, gamma_adj)
            reasons.append(f'Gamma exposure ({gamma_exposure:.2f}) exceeds limit ({gamma_limit:.2f})')
            
        # Check vega exposure
        vega_limit = self.portfolio_value * self.max_vega_exposure
        if vega_exposure > vega_limit:
            vega_adj = vega_limit / vega_exposure
            adjustment_factor = min(adjustment_factor, vega_adj)
            reasons.append(f'Vega exposure ({vega_exposure:.2f}) exceeds limit ({vega_limit:.2f})')
            
        # Check theta decay
        theta_limit = self.portfolio_value * self.max_theta_decay
        if theta_exposure > theta_limit:
            theta_adj = theta_limit / theta_exposure
            adjustment_factor = min(adjustment_factor, theta_adj)
            reasons.append(f'Theta decay ({theta_exposure:.2f}) exceeds limit ({theta_limit:.2f})')
            
        # Check delta exposure
        if delta_exposure > 0:
            hedge_ratio = greeks_data.get('hedge_ratio', 0)
            if hedge_ratio < self.min_delta_hedge:
                delta_adj = self.min_delta_hedge / max(hedge_ratio, 0.01)
                adjustment_factor = min(adjustment_factor, delta_adj)
                reasons.append(f'Delta hedge ratio ({hedge_ratio:.2f}) below minimum ({self.min_delta_hedge:.2f})')
                
        adjusted_size = position_size * adjustment_factor
        
        return adjusted_size, {
            'adjustment_factor': adjustment_factor,
            'reasons': reasons,
            'exposures': {
                'gamma': gamma_exposure,
                'vega': vega_exposure,
                'theta': theta_exposure,
                'delta': delta_exposure
            }
        }
        
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr[-period:])
        return float(atr)
        
    def calculate_position_size(
            self,
            current_price: float,
            volatility: float,
            market_regime: str,
            greeks_data: Optional[Dict[str, float]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        vol_factor = 1.0 - (volatility / 100)
        regime_factors = {'trending': 1.0, 'ranging': 0.7, 'high_volatility': 0.5}
        regime_factor = regime_factors.get(market_regime, 0.5)
        
        base_position = float(self.portfolio_value * self.config.max_position_size)
        adjusted_position = float(base_position * vol_factor * regime_factor)
        
        risk_amount = float(self.portfolio_value * self.config.max_risk_per_trade)
        max_shares = float(risk_amount / (current_price * volatility))
        
        final_position = float(min(adjusted_position, max_shares * current_price))
        shares = int(final_position / current_price)
        
        metrics: Dict[str, Any] = {
            'position_value': float(shares * current_price),
            'position_size_pct': float((shares * current_price) / self.portfolio_value),
            'risk_amount': float(risk_amount),
            'volatility_factor': float(vol_factor),
            'regime_factor': float(regime_factor)
        }
        
        # Adjust position based on Greeks if available
        if greeks_data:
            adjusted_position, greek_metrics = self.adjust_for_greeks(final_position, greeks_data)
            shares = int(adjusted_position / current_price)
            metrics['position_value'] = float(shares * current_price)
            metrics['position_size_pct'] = float((shares * current_price) / self.portfolio_value)
            metrics['greek_adjustment'] = greek_metrics
            
        return shares, metrics
    
    def calculate_stop_loss(
            self,
            entry_price: float,
            position_type: str,
            atr: float
    ) -> Tuple[float, float]:
        atr_stop = atr * self.config.atr_multiplier
        
        if position_type == 'long':
            stop_loss = entry_price - atr_stop
            profit_target = entry_price + (atr_stop * 1.5)
        else:
            stop_loss = entry_price + atr_stop
            profit_target = entry_price - (atr_stop * 1.5)
            
        return stop_loss, profit_target
    
    def update_trailing_stop(
            self,
            position_type: str,
            current_price: float,
            entry_price: float,
            highest_price: float,
            lowest_price: float,
            current_stop: float
    ) -> Optional[float]:
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
    
    def validate_trade(
            self,
            entry_price: float,
            stop_loss: float,
            profit_target: float,
            position_type: str,
            greeks_data: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
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
            
        # Validate Greeks exposure if data is available
        if greeks_data:
            _, greek_metrics = self.adjust_for_greeks(self.portfolio_value * self.config.max_position_size, greeks_data)
            if greek_metrics['adjustment_factor'] < 0.5:  # If Greeks require more than 50% position reduction
                return False, f"Excessive Greeks exposure: {'; '.join(greek_metrics['reasons'])}"
            
        return True, "Trade validated"
