import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StressScenario:
    name: str
    price_shock: float
    volatility_multiplier: float
    volume_multiplier: float
    correlation_change: float
    liquidity_shock: float

@dataclass
class StressTestResult:
    scenario_name: str
    max_drawdown: float
    recovery_time: int
    var_95: float
    sharpe_ratio: float
    liquidity_score: float
    survival_score: float

class StressTester:
    def __init__(self,
                 base_scenarios: Optional[List[StressScenario]] = None):
        self.scenarios = base_scenarios or self._generate_base_scenarios()
        
    def run_stress_tests(self,
                        price_data: pd.DataFrame,
                        position_sizes: np.ndarray,
                        risk_limits: Dict[str, float]) -> List[StressTestResult]:
        results = []
        
        for scenario in self.scenarios:
            stressed_data = self._apply_stress_scenario(price_data, scenario)
            result = self._evaluate_strategy(
                stressed_data,
                position_sizes,
                risk_limits,
                scenario
            )
            results.append(result)
            
        return results
        
    def _generate_base_scenarios(self) -> List[StressScenario]:
        return [
            StressScenario(
                name="Market Crash",
                price_shock=-0.15,
                volatility_multiplier=3.0,
                volume_multiplier=2.5,
                correlation_change=0.3,
                liquidity_shock=-0.4
            ),
            StressScenario(
                name="Flash Crash",
                price_shock=-0.08,
                volatility_multiplier=5.0,
                volume_multiplier=3.0,
                correlation_change=0.5,
                liquidity_shock=-0.6
            ),
            StressScenario(
                name="Sector Rotation",
                price_shock=-0.05,
                volatility_multiplier=2.0,
                volume_multiplier=1.5,
                correlation_change=-0.2,
                liquidity_shock=-0.2
            ),
            StressScenario(
                name="Liquidity Crisis",
                price_shock=-0.10,
                volatility_multiplier=2.5,
                volume_multiplier=0.3,
                correlation_change=0.4,
                liquidity_shock=-0.8
            ),
            StressScenario(
                name="Volatility Spike",
                price_shock=-0.03,
                volatility_multiplier=4.0,
                volume_multiplier=2.0,
                correlation_change=0.1,
                liquidity_shock=-0.3
            )
        ]
        
    def _apply_stress_scenario(self,
                             data: pd.DataFrame,
                             scenario: StressScenario) -> pd.DataFrame:
        stressed_data = data.copy()
        
        # Apply price shock
        stressed_data['Close'] *= (1 + scenario.price_shock)
        
        # Apply volatility shock
        returns = stressed_data['Close'].pct_change()
        stressed_returns = returns * scenario.volatility_multiplier
        stressed_data['Close'] = stressed_data['Close'].iloc[0] * \
                                (1 + stressed_returns).cumprod()
        
        # Apply volume shock
        stressed_data['Volume'] *= scenario.volume_multiplier
        
        # Apply correlation effects
        if 'Sector' in stressed_data.columns:
            sector_returns = stressed_data.groupby('Sector')['Close'].pct_change()
            sector_returns *= (1 + scenario.correlation_change)
            
        # Apply liquidity shock
        if 'Bid-Ask' in stressed_data.columns:
            stressed_data['Bid-Ask'] *= (1 - scenario.liquidity_shock)
            
        return stressed_data
        
    def _evaluate_strategy(self,
                          stressed_data: pd.DataFrame,
                          position_sizes: np.ndarray,
                          risk_limits: Dict[str, float],
                          scenario: StressScenario) -> StressTestResult:
        # Calculate returns under stress
        returns = stressed_data['Close'].pct_change()
        position_returns = returns * position_sizes
        
        # Calculate drawdown
        cumulative_returns = (1 + position_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate recovery time
        if max_drawdown < 0:
            underwater = drawdowns < 0
            recovery_periods = underwater.value_counts()
            recovery_time = recovery_periods.get(True, 0)
        else:
            recovery_time = 0
            
        # Calculate risk metrics
        var_95 = np.percentile(position_returns, 5)
        excess_returns = position_returns - 0.02/252  # Assuming 2% risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate liquidity score
        if 'Bid-Ask' in stressed_data.columns:
            liquidity_score = 1.0 / (1.0 + stressed_data['Bid-Ask'].mean())
        else:
            liquidity_score = 0.5
            
        # Calculate survival score
        risk_breaches = 0
        if max_drawdown < -risk_limits.get('max_drawdown', 0.15):
            risk_breaches += 1
        if var_95 < -risk_limits.get('var_limit', 0.10):
            risk_breaches += 1
            
        survival_score = 1.0 - (risk_breaches / 2)
        
        return StressTestResult(
            scenario_name=scenario.name,
            max_drawdown=float(max_drawdown),
            recovery_time=int(recovery_time),
            var_95=float(var_95),
            sharpe_ratio=float(sharpe),
            liquidity_score=float(liquidity_score),
            survival_score=float(survival_score)
        )
