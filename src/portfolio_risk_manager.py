import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

@dataclass
class PortfolioConfig:
    max_portfolio_risk: float = 0.15  # Maximum portfolio-wide risk (15%)
    sector_exposure_limit: float = 0.30  # Maximum exposure to a single sector (30%)
    correlation_threshold: float = 0.7  # Correlation threshold for risk aggregation
    max_leverage: float = 1.5  # Maximum portfolio leverage
    min_diversification: int = 3  # Minimum number of positions

class PortfolioRiskManager:
    def __init__(self, initial_portfolio_value: float):
        self.portfolio_value = initial_portfolio_value
        self.config = PortfolioConfig()
        self.positions: Dict[str, Dict] = {}
        self.sector_exposure: Dict[str, float] = defaultdict(float)
        self.correlations: Dict[str, Dict[str, float]] = {}
        
    def add_position(self, symbol: str, value: float, sector: str, 
                    beta: float, correlation_data: Dict[str, float]) -> None:
        self.positions[symbol] = {
            'value': value,
            'sector': sector,
            'beta': beta,
            'weight': value / self.portfolio_value
        }
        self.sector_exposure[sector] += value / self.portfolio_value
        self.correlations[symbol] = correlation_data
        
    def remove_position(self, symbol: str) -> None:
        if symbol in self.positions:
            position = self.positions[symbol]
            self.sector_exposure[position['sector']] -= position['weight']
            del self.positions[symbol]
            del self.correlations[symbol]
            
    def calculate_portfolio_risk(self) -> Tuple[float, Dict[str, float]]:
        if not self.positions:
            return 0.0, {'diversification': 0, 'concentration': 0, 'correlation': 0}
            
        weights = np.array([pos['weight'] for pos in self.positions.values()])
        betas = np.array([pos['beta'] for pos in self.positions.values()])
        
        # Calculate portfolio beta
        portfolio_beta = np.sum(weights * betas)
        
        # Calculate concentration risk
        herfindahl = np.sum(weights ** 2)
        concentration_risk = herfindahl * portfolio_beta
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk()
        
        # Calculate diversification score
        diversification = 1 / (herfindahl * len(self.positions))
        
        total_risk = (concentration_risk + correlation_risk) * (2 - diversification)
        
        metrics = {
            'diversification': diversification,
            'concentration': concentration_risk,
            'correlation': correlation_risk
        }
        
        return total_risk, metrics
        
    def _calculate_correlation_risk(self) -> float:
        if len(self.positions) < 2:
            return 0.0
            
        total_correlation = 0.0
        count = 0
        
        symbols = list(self.positions.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                if sym2 in self.correlations[sym1]:
                    correlation = abs(self.correlations[sym1][sym2])
                    if correlation > self.config.correlation_threshold:
                        total_correlation += correlation
                        count += 1
                        
        return total_correlation / max(1, count)
        
    def validate_new_position(self, symbol: str, value: float, sector: str,
                            beta: float, correlation_data: Dict[str, float]) -> Tuple[bool, str, Dict]:
        # Temporarily add the position
        self.add_position(symbol, value, sector, beta, correlation_data)
        
        validation_metrics = {}
        
        # Check sector exposure
        sector_exposure = self.sector_exposure[sector]
        validation_metrics['sector_exposure'] = sector_exposure
        
        if sector_exposure > self.config.sector_exposure_limit:
            self.remove_position(symbol)
            return False, "Sector exposure limit exceeded", validation_metrics
            
        # Check portfolio risk
        portfolio_risk, risk_metrics = self.calculate_portfolio_risk()
        validation_metrics.update(risk_metrics)
        
        if portfolio_risk > self.config.max_portfolio_risk:
            self.remove_position(symbol)
            return False, "Portfolio risk limit exceeded", validation_metrics
            
        # Check leverage
        total_exposure = sum(pos['value'] for pos in self.positions.values())
        leverage = total_exposure / self.portfolio_value
        validation_metrics['leverage'] = leverage
        
        if leverage > self.config.max_leverage:
            self.remove_position(symbol)
            return False, "Leverage limit exceeded", validation_metrics
            
        # Check diversification
        if len(self.positions) < self.config.min_diversification:
            validation_metrics['diversification_warning'] = True
            
        # Remove the temporary position
        self.remove_position(symbol)
        
        return True, "Position validated", validation_metrics
        
    def get_position_size_adjustment(self, portfolio_risk: float) -> float:
        if portfolio_risk >= self.config.max_portfolio_risk:
            return 0.0
        elif portfolio_risk >= self.config.max_portfolio_risk * 0.8:
            return 1.0 - (portfolio_risk / self.config.max_portfolio_risk)
        return 1.0
