from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import requests

@dataclass
class DarkPoolData:
    timestamp: datetime
    volume: float
    price: float
    side: str  # 'buy' or 'sell'
    block_size: float

class DarkPoolAnalyzer:
    def __init__(self, polygon_key: str):
        self.api_key = polygon_key
        self.base_url = "https://api.polygon.io/v2"
        
    def fetch_dark_pool_data(self, symbol: str, days: int = 5) -> List[DarkPoolData]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "apiKey": self.api_key,
            "limit": 50000,
            "timestamp.gte": start_date.strftime("%Y-%m-%d"),
            "timestamp.lte": end_date.strftime("%Y-%m-%d")
        }
        
        endpoint = f"{self.base_url}/ticks/stocks/nbbo/{symbol}"
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        dark_pool_trades = []
        for trade in data.get('results', []):
            if trade.get('conditions', []).intersection({'7', '4', 'D'}):  # Dark pool conditions
                dark_pool_trades.append(DarkPoolData(
                    timestamp=datetime.fromtimestamp(trade['t'] / 1000),
                    volume=float(trade['s']),
                    price=float(trade['p']),
                    side='buy' if trade.get('q', 0) > 0 else 'sell',
                    block_size=float(trade['s']) * float(trade['p'])
                ))
                
        return sorted(dark_pool_trades, key=lambda x: x.timestamp, reverse=True)
        
    def analyze_dark_pool_activity(self, trades: List[DarkPoolData]) -> Dict[str, float]:
        if not trades:
            return {
                'net_flow': 0.0,
                'buy_pressure': 0.0,
                'large_trade_impact': 0.0,
                'volume_distribution': 0.0,
                'composite_score': 0.0
            }
            
        total_volume = sum(trade.volume for trade in trades)
        buy_volume = sum(trade.volume for trade in trades if trade.side == 'buy')
        sell_volume = total_volume - buy_volume
        
        # Net flow (-1 to 1)
        net_flow = float((buy_volume - sell_volume) / total_volume if total_volume > 0 else 0)
        
        # Buy pressure (0 to 1)
        buy_pressure = float(buy_volume / total_volume if total_volume > 0 else 0)
        
        # Large trade impact
        avg_trade_size = float(total_volume / len(trades))
        large_trades = [t for t in trades if t.volume > avg_trade_size * 2]
        large_trade_volume = sum(t.volume for t in large_trades)
        large_trade_impact = float(large_trade_volume / total_volume if total_volume > 0 else 0)
        
        # Volume distribution analysis
        volumes = [float(t.volume) for t in trades]
        volume_std = float(np.std(volumes) if volumes else 0)
        volume_mean = float(np.mean(volumes) if volumes else 0)
        volume_distribution = float(volume_std / volume_mean if volume_mean > 0 else 0)
        
        # Composite score (-1 to 1)
        composite_score = (
            0.4 * net_flow +
            0.3 * (buy_pressure - 0.5) * 2 +  # Convert to -1 to 1 range
            0.2 * (large_trade_impact - 0.5) * 2 +
            0.1 * (volume_distribution - 0.5) * 2
        )
        
        return {
            'net_flow': net_flow,
            'buy_pressure': buy_pressure,
            'large_trade_impact': large_trade_impact,
            'volume_distribution': volume_distribution,
            'composite_score': composite_score
        }
        
    def get_significant_levels(self, trades: List[DarkPoolData]) -> List[Dict[str, float]]:
        if not trades:
            return []
            
        # Group trades by price levels
        price_levels = {}
        for trade in trades:
            price = round(trade.price, 2)
            if price not in price_levels:
                price_levels[price] = 0
            price_levels[price] += trade.volume
            
        # Find significant levels (local maxima in volume)
        significant = []
        prices = sorted(price_levels.keys())
        for i in range(1, len(prices) - 1):
            if (price_levels[prices[i]] > price_levels[prices[i-1]] and 
                price_levels[prices[i]] > price_levels[prices[i+1]]):
                significant.append({
                    'price': prices[i],
                    'volume': price_levels[prices[i]],
                    'strength': price_levels[prices[i]] / sum(price_levels.values())
                })
                
        return sorted(significant, key=lambda x: x['strength'], reverse=True)
