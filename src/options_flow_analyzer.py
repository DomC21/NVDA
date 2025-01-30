import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import requests

@dataclass
class OptionsFlow:
    call_volume: float
    put_volume: float
    call_oi: float
    put_oi: float
    pcr_volume: float
    pcr_oi: float
    unusual_activity: List[Dict]
    institutional_sentiment: float

class OptionsFlowAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.unusualwhales.com/api/v1"
        
    def fetch_options_flow(self, symbol: str) -> OptionsFlow:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        endpoint = f"{self.base_url}/options/flow/{symbol}"
        
        response = requests.get(endpoint, headers=headers)
        data = response.json()
        
        unusual_activity = self._process_unusual_activity(data.get('flow', []))
        
        return OptionsFlow(
            call_volume=data.get('call_volume', 0),
            put_volume=data.get('put_volume', 0),
            call_oi=data.get('call_open_interest', 0),
            put_oi=data.get('put_open_interest', 0),
            pcr_volume=data.get('put_call_ratio_volume', 0),
            pcr_oi=data.get('put_call_ratio_oi', 0),
            unusual_activity=unusual_activity,
            institutional_sentiment=self._calculate_institutional_sentiment(data)
        )
        
    def _process_unusual_activity(self, flow_data: List[Dict]) -> List[Dict]:
        processed = []
        for flow in flow_data:
            if flow.get('premium', 0) >= 100000:  # Focus on large trades
                processed.append({
                    'strike': flow.get('strike'),
                    'expiration': flow.get('expiration'),
                    'type': flow.get('type'),
                    'premium': flow.get('premium'),
                    'volume': flow.get('volume'),
                    'open_interest': flow.get('open_interest'),
                    'is_opening': flow.get('is_opening', False)
                })
        return processed
        
    def _calculate_institutional_sentiment(self, data: Dict) -> float:
        call_premium = sum(flow.get('premium', 0) for flow in data.get('flow', [])
                         if flow.get('type') == 'call' and flow.get('premium', 0) >= 100000)
        put_premium = sum(flow.get('premium', 0) for flow in data.get('flow', [])
                        if flow.get('type') == 'put' and flow.get('premium', 0) >= 100000)
        
        total_premium = call_premium + put_premium
        if total_premium == 0:
            return 0.0
            
        return (call_premium - put_premium) / total_premium
        
    def analyze_flow(self, flow: OptionsFlow) -> Dict[str, float]:
        analysis = {}
        
        # Volume-based signals
        analysis['volume_signal'] = -1 if flow.pcr_volume > 1.5 else 1 if flow.pcr_volume < 0.7 else 0
        
        # Open Interest signals
        analysis['oi_signal'] = -1 if flow.pcr_oi > 1.3 else 1 if flow.pcr_oi < 0.8 else 0
        
        # Institutional activity
        analysis['institutional_signal'] = flow.institutional_sentiment
        
        # Unusual activity significance
        total_unusual_premium = sum(trade['premium'] for trade in flow.unusual_activity)
        analysis['unusual_activity_score'] = min(1.0, total_unusual_premium / 10000000)
        
        # Composite score (-1 to 1)
        weights = {
            'volume_signal': 0.2,
            'oi_signal': 0.2,
            'institutional_signal': 0.4,
            'unusual_activity_score': 0.2
        }
        
        analysis['composite_score'] = sum(score * weights[signal] 
                                        for signal, score in analysis.items())
        
        return analysis
