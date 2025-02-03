from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd


@dataclass
class UnusualWhalesAnalysis:
    dark_pool_metrics: Dict[str, float]
    option_flow_metrics: Dict[str, float]
    market_tide_score: float
    greek_exposure: Dict[str, float]
    volume_levels: Dict[float, float]
    signal_strength: float


class UnusualWhalesAnalyzer:
    def __init__(self, min_block_size: float = 100000):
        self.min_block_size = min_block_size
        
    def analyze_dark_pool(self, data: pd.DataFrame) -> Dict[str, float]:
        if data.empty:
            return {
                'net_flow': 0.0,
                'block_trade_ratio': 0.0,
                'price_impact': 0.0,
                'volume_concentration': 0.0
            }
        total_volume = data['volume'].sum()
        block_volume = data[data['volume'] >= self.min_block_size]['volume'].sum()
        
        # Calculate volume-weighted average price
        vwap = (
            (data['price'] * data['volume']).sum() / total_volume if total_volume > 0 else 0
        )
        
        # Calculate price impact
        price_std = data['price'].std()
        price_impact = price_std / vwap if vwap > 0 else 0
        
        # Volume concentration at price levels
        volume_by_price = data.groupby('price')['volume'].sum()
        max_volume = volume_by_price.max()
        volume_concentration = max_volume / total_volume if total_volume > 0 else 0
        
        return {
            'net_flow': (
                block_volume / total_volume if total_volume > 0 else 0
            ),
            'block_trade_ratio': (
                len(data[data['volume'] >= self.min_block_size]) / len(data)
                if len(data) > 0 else 0
            ),
            'price_impact': price_impact,
            'volume_concentration': volume_concentration
        }
        
    def analyze_option_flow(self, data: pd.DataFrame) -> Dict[str, float]:
        if data.empty:
            return {
                'call_put_ratio': 1.0,
                'premium_ratio': 0.0,
                'large_order_ratio': 0.0,
                'bullish_flow_score': 0.0
            }
            
        call_volume = data[data['type'] == 'call']['volume'].sum()
        put_volume = data[data['type'] == 'put']['volume'].sum()
        
        call_premium = data[data['type'] == 'call']['premium'].sum()
        put_premium = data[data['type'] == 'put']['premium'].sum()
        
        total_orders = len(data)
        avg_premium = data['premium'].mean()
        large_orders = len(data[data['premium'] > avg_premium * 2])
        
        return {
            'call_put_ratio': (call_volume / put_volume 
                           if put_volume > 0 else float('inf')),
            'premium_ratio': (call_premium / put_premium 
                            if put_premium > 0 else float('inf')),
            'large_order_ratio': (large_orders / total_orders 
                                if total_orders > 0 else 0),
            'bullish_flow_score': (
                (call_premium - put_premium) / (call_premium + put_premium)
                if (call_premium + put_premium) > 0 else 0
            )
        }
        
    def analyze_market_tide(self, data: Dict) -> float:
        if not data or 'score' not in data:
            return 0.5
        return float(data['score'])
        
    def analyze_greeks(self, data: Dict) -> Dict[str, float]:
        if not data:
            return {
                'net_gamma': 0.0,
                'net_delta': 0.0,
                'net_theta': 0.0,
                'net_vega': 0.0
            }
        return {
            'net_gamma': data.get('gamma', 0.0),
            'net_delta': data.get('delta', 0.0),
            'net_theta': data.get('theta', 0.0),
            'net_vega': data.get('vega', 0.0)
        }
        
    def analyze_volume_levels(self, data: pd.DataFrame) -> Dict[float, float]:
        if data.empty:
            return {}
        return data.groupby('strike')['volume'].sum().to_dict()
        
    def generate_signal(self, analysis: UnusualWhalesAnalysis) -> Tuple[str, float]:
        """Generate trading signal based on combined analysis metrics.
        
        Returns:
            Tuple[str, float]: Direction ('buy' or 'sell') and signal strength (0.0 to 1.0)
        """
        bullish_factors: int = 0
        total_factors: int = 0
        
        # Dark pool analysis
        if analysis.dark_pool_metrics.get('net_flow', 0.0) > 0.6:
            bullish_factors += 1
        elif analysis.dark_pool_metrics.get('net_flow', 0.0) < 0.4:
            bullish_factors -= 1
        total_factors += 1
        
        # Options flow
        if analysis.option_flow_metrics.get('bullish_flow_score', 0.0) > 0.2:
            bullish_factors += 1
        elif analysis.option_flow_metrics.get('bullish_flow_score', 0.0) < -0.2:
            bullish_factors -= 1
        total_factors += 1
        
        # Market tide
        if analysis.market_tide_score > 0.6:
            bullish_factors += 1
        elif analysis.market_tide_score < 0.4:
            bullish_factors -= 1
        total_factors += 1
        
        # Greek exposure
        if analysis.greek_exposure.get('net_delta', 0.0) > 0:
            bullish_factors += 1
        elif analysis.greek_exposure.get('net_delta', 0.0) < 0:
            bullish_factors -= 1
        total_factors += 1
        
        signal_strength = float(abs(bullish_factors) / total_factors if total_factors > 0 else 0.0)
        direction = 'buy' if bullish_factors > 0 else 'sell'
        
        return direction, signal_strength
