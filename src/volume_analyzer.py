import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class VolumeProfile:
    price_levels: np.ndarray
    volume_at_price: np.ndarray
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_weighted_price: float

class VolumeAnalyzer:
    def __init__(self, price_data: pd.DataFrame):
        self.data = price_data
        self.volume_profile = None
        
    def calculate_volume_profile(self, num_bins: int = 50, 
                               value_area_pct: float = 0.70) -> VolumeProfile:
        prices = np.array(self.data['Close'].values)
        volumes = np.array(self.data['Volume'].values)
        
        price_range = np.linspace(np.min(prices), np.max(prices), num_bins)
        volume_at_price = np.zeros(num_bins - 1)
        
        for i in range(len(price_range) - 1):
            mask = (prices >= price_range[i]) & (prices < price_range[i + 1])
            volume_at_price[i] = volumes[mask].sum()
            
        poc_index = np.argmax(volume_at_price)
        poc_price = (price_range[poc_index] + price_range[poc_index + 1]) / 2
        
        total_volume = volume_at_price.sum()
        target_volume = total_volume * value_area_pct
        cumulative_volume = 0
        left_index = right_index = poc_index
        
        while cumulative_volume < target_volume and (left_index > 0 or right_index < len(volume_at_price) - 1):
            left_vol = volume_at_price[left_index - 1] if left_index > 0 else 0
            right_vol = volume_at_price[right_index + 1] if right_index < len(volume_at_price) - 1 else 0
            
            if left_vol > right_vol and left_index > 0:
                left_index -= 1
                cumulative_volume += left_vol
            elif right_index < len(volume_at_price) - 1:
                right_index += 1
                cumulative_volume += right_vol
                
        value_area_high = price_range[right_index + 1]
        value_area_low = price_range[left_index]
        
        vwap = float(np.sum(prices * volumes) / np.sum(volumes))
        
        self.volume_profile = VolumeProfile(
            price_levels=price_range[:-1],
            volume_at_price=volume_at_price,
            poc_price=poc_price,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            volume_weighted_price=vwap
        )
        
        return self.volume_profile
        
    def get_support_resistance(self, min_volume_threshold: float = 0.1) -> List[float]:
        volume_profile = self.volume_profile
        if volume_profile is None:
            volume_profile = self.calculate_volume_profile()
            
        volume_threshold = np.max(volume_profile.volume_at_price) * min_volume_threshold
        significant_levels = []
        
        for i in range(1, len(volume_profile.volume_at_price) - 1):
            if volume_profile.volume_at_price[i] > volume_threshold:
                if (volume_profile.volume_at_price[i] > volume_profile.volume_at_price[i-1] and
                    volume_profile.volume_at_price[i] > volume_profile.volume_at_price[i+1]):
                    significant_levels.append(volume_profile.price_levels[i])
                    
        return significant_levels
        
    def analyze_volume_trend(self, window: int = 20) -> Dict[str, float]:
        volume = self.data['Volume']
        price = self.data['Close']
        
        avg_volume = volume.rolling(window=window).mean()
        volume_ratio = volume / avg_volume
        
        price_change = price.pct_change()
        volume_momentum = volume_ratio * price_change
        
        return {
            'volume_ratio': float(volume_ratio.iloc[-1]),
            'volume_momentum': float(volume_momentum.iloc[-1]),
            'trend_strength': float(abs(volume_momentum.iloc[-1]))
        }
        
    def get_entry_exit_signals(self, current_price: float) -> Dict[str, Dict[str, float]]:
        volume_profile = self.volume_profile
        if volume_profile is None:
            volume_profile = self.calculate_volume_profile()
            
        volume_trend = self.analyze_volume_trend()
        support_resistance = self.get_support_resistance()
        
        supports = [level for level in support_resistance if level < current_price]
        resistances = [level for level in support_resistance if level > current_price]
        
        nearest_support = max(supports) if supports else volume_profile.value_area_low
        nearest_resistance = min(resistances) if resistances else volume_profile.value_area_high
        
        long_risk = current_price - nearest_support
        long_reward = nearest_resistance - current_price
        long_rr_ratio = long_reward / long_risk if long_risk > 0 else 0
        
        short_risk = nearest_resistance - current_price
        short_reward = current_price - nearest_support
        short_rr_ratio = short_reward / short_risk if short_risk > 0 else 0
        
        return {
            'long': {
                'entry_price': current_price,
                'stop_loss': nearest_support,
                'take_profit': nearest_resistance,
                'risk_reward_ratio': long_rr_ratio,
                'volume_confidence': volume_trend['volume_ratio'] if volume_trend['volume_momentum'] > 0 else 0
            },
            'short': {
                'entry_price': current_price,
                'stop_loss': nearest_resistance,
                'take_profit': nearest_support,
                'risk_reward_ratio': short_rr_ratio,
                'volume_confidence': volume_trend['volume_ratio'] if volume_trend['volume_momentum'] < 0 else 0
            }
        }
