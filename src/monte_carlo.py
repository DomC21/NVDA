import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class SimulationResult:
    returns: np.ndarray
    drawdowns: np.ndarray
    sharpe_ratio: float
    var_95: float
    cvar_95: float

class MonteCarloSimulator:
    def __init__(self, 
                 n_simulations: int = 1000,
                 time_horizon: int = 252,
                 confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.confidence_level = confidence_level
        
    def simulate_returns(self,
                        historical_returns: np.ndarray,
                        initial_value: float = 100000.0) -> SimulationResult:
        mu = np.mean(historical_returns)
        sigma = np.std(historical_returns)
        
        # Generate random returns using geometric Brownian motion
        dt = 1/252  # Daily timestep
        returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt,
            sigma * np.sqrt(dt),
            size=(self.n_simulations, self.time_horizon)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.exp(np.cumsum(returns, axis=1))
        portfolio_values = initial_value * cumulative_returns
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(portfolio_values, axis=1)
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Calculate risk metrics
        final_values = portfolio_values[:, -1]
        total_returns = (final_values - initial_value) / initial_value
        
        sharpe = self._calculate_sharpe_ratio(total_returns, 0.02)  # Assuming 2% risk-free rate
        var_95 = self._calculate_var(total_returns)
        cvar_95 = self._calculate_cvar(total_returns)
        
        return SimulationResult(
            returns=total_returns,
            drawdowns=drawdowns,
            sharpe_ratio=sharpe,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
    def assess_strategy_risk(self,
                           historical_returns: np.ndarray,
                           position_sizes: np.ndarray,
                           stop_losses: np.ndarray) -> Dict[str, float]:
        results = []
        
        for _ in range(self.n_simulations):
            # Randomly sample historical returns
            sampled_returns = np.random.choice(
                historical_returns,
                size=self.time_horizon,
                replace=True
            )
            
            # Apply position sizing and stop losses
            adjusted_returns = self._apply_risk_controls(
                sampled_returns,
                position_sizes,
                stop_losses
            )
            
            results.append(self._calculate_path_metrics(adjusted_returns))
            
        return self._aggregate_results(results)
        
    def _apply_risk_controls(self,
                            returns: np.ndarray,
                            position_sizes: np.ndarray,
                            stop_losses: np.ndarray) -> np.ndarray:
        adjusted_returns = returns * position_sizes
        cumulative_returns = np.cumprod(1 + adjusted_returns)
        
        # Apply stop losses
        for i in range(len(returns)):
            if cumulative_returns[i] < (1 - stop_losses[i]):
                adjusted_returns[i:] = 0
                break
                
        return adjusted_returns
        
    def _calculate_path_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        cumulative_return = np.prod(1 + returns) - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        volatility = np.std(returns) * np.sqrt(252)
        
        return {
            'return': cumulative_return,
            'drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe': self._calculate_sharpe_ratio(returns, 0.02)
        }
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return np.min(drawdowns)
        
    def _calculate_sharpe_ratio(self,
                              returns: np.ndarray,
                              risk_free_rate: float) -> float:
        excess_returns = np.mean(returns) - risk_free_rate
        return excess_returns / (np.std(returns) * np.sqrt(252))
        
    def _calculate_var(self, returns: np.ndarray) -> float:
        return np.percentile(returns, (1 - self.confidence_level) * 100)
        
    def _calculate_cvar(self, returns: np.ndarray) -> float:
        var = self._calculate_var(returns)
        return np.mean(returns[returns <= var])
        
    def _aggregate_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        metrics = pd.DataFrame(results)
        
        return {
            'expected_return': float(metrics['return'].mean()),
            'expected_drawdown': float(metrics['drawdown'].mean()),
            'var_95': float(np.percentile(metrics['return'], 5)),
            'cvar_95': float(metrics[metrics['return'] <= metrics['return'].quantile(0.05)]['return'].mean()),
            'sharpe_ratio': float(metrics['sharpe'].mean()),
            'success_rate': float((metrics['return'] > 0).mean())
        }
