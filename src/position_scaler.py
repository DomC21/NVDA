import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ScalingFactors:
    confidence: float
    volatility: float
    liquidity: float
    market_impact: float
    cost_efficiency: float

class PositionScaler:
    def __init__(self, 
                 base_position_size: float,
                 max_position_size: float,
                 min_position_size: float):
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        
    def calculate_position_size(self,
                              signal_confidence: float,
                              volatility: float,
                              avg_volume: float,
                              current_price: float,
                              transaction_costs: Dict) -> Dict[str, float]:
        scaling_factors = self._compute_scaling_factors(
            signal_confidence, volatility, avg_volume,
            current_price, transaction_costs
        )
        
        composite_scale = (
            scaling_factors.confidence * 0.3 +
            scaling_factors.volatility * 0.2 +
            scaling_factors.liquidity * 0.2 +
            scaling_factors.market_impact * 0.15 +
            scaling_factors.cost_efficiency * 0.15
        )
        
        scaled_size = self.base_position_size * composite_scale
        final_size = np.clip(scaled_size, self.min_position_size, self.max_position_size)
        
        entry_size = final_size * 0.6  # Initial entry with 60%
        reserve_size = final_size * 0.4  # Reserve 40% for scaling
        
        return {
            'initial_entry': entry_size,
            'reserve': reserve_size,
            'total_size': final_size,
            'scaling_factors': {
                'confidence': scaling_factors.confidence,
                'volatility': scaling_factors.volatility,
                'liquidity': scaling_factors.liquidity,
                'market_impact': scaling_factors.market_impact,
                'cost_efficiency': scaling_factors.cost_efficiency
            }
        }
        
    def _compute_scaling_factors(self,
                               signal_confidence: float,
                               volatility: float,
                               avg_volume: float,
                               current_price: float,
                               transaction_costs: Dict) -> ScalingFactors:
        confidence_factor = min(1.0, signal_confidence / 100)
        
        volatility_factor = 1.0 - np.clip(volatility * 10, 0, 0.5)
        
        position_value = self.base_position_size * current_price
        liquidity_factor = min(1.0, (avg_volume * current_price) / (position_value * 20))
        
        market_impact = transaction_costs.get('market_impact', 0)
        impact_factor = 1.0 - np.clip(market_impact / (position_value * 0.001), 0, 0.5)
        
        total_cost = transaction_costs.get('total_cost', 0)
        cost_factor = 1.0 - np.clip(total_cost / (position_value * 0.002), 0, 0.5)
        
        return ScalingFactors(
            confidence=confidence_factor,
            volatility=volatility_factor,
            liquidity=liquidity_factor,
            market_impact=impact_factor,
            cost_efficiency=cost_factor
        )
