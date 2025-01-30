import numpy as np
from typing import Dict, Optional

class SlippageModel:
    def __init__(self, base_spread: float = 0.0001, 
                 volume_impact: float = 0.1,
                 volatility_impact: float = 0.2):
        self.base_spread = base_spread
        self.volume_impact = volume_impact
        self.volatility_impact = volatility_impact
        
    def estimate_transaction_cost(self, price: float, size: float, 
                                market_data: Dict[str, float]) -> Dict[str, float]:
        spread_cost = self.base_spread * price * size
        
        volume_ratio = size / market_data['avg_volume'] if market_data['avg_volume'] > 0 else 1
        market_impact = price * size * self.volume_impact * np.log1p(volume_ratio)
        
        volatility_cost = price * size * self.volatility_impact * market_data['volatility']
        
        slippage = market_impact + volatility_cost
        commission = max(1.0, size * price * 0.0001)
        
        total_cost = spread_cost + slippage + commission
        
        return {
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'volatility_cost': volatility_cost,
            'commission': commission,
            'slippage_cost': slippage,
            'total_cost': total_cost
        }
        
    def optimize_order_size(self, target_size: float, max_slippage: float,
                          market_data: Dict[str, float]) -> float:
        min_size = 1.0
        max_size = target_size
        optimal_size = target_size
        
        while min_size <= max_size:
            current_size = (min_size + max_size) / 2
            costs = self.estimate_transaction_cost(
                price=market_data['price'],
                size=current_size,
                market_data=market_data
            )
            
            slippage_ratio = costs['slippage_cost'] / (market_data['price'] * current_size)
            
            if abs(slippage_ratio - max_slippage) < 0.0001:
                return current_size
            elif slippage_ratio > max_slippage:
                max_size = current_size - 1
            else:
                min_size = current_size + 1
                optimal_size = current_size
                
        return optimal_size
