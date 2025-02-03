import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from data_collector import DataCollector
from price_predictor import PricePredictor
from risk_manager import RiskManager
from market_regime_analyzer import MarketRegimeAnalyzer
from analysis import StockAnalyzer
from config import SYMBOL

class BacktestRunner:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = 0
        self.trades = []
        self.collector = DataCollector()
        self.predictor = PricePredictor()
        self.risk_manager = RiskManager(initial_capital)
        self.market_analyzer = MarketRegimeAnalyzer()
        self.stock_analyzer = StockAnalyzer()
        self.model_trained = False
        
    def run_backtest(self, start_date, end_date):
        data = self.collector.collect_historical_data('NVDA', start_date, end_date)
        if data is None or len(data) < 50:
            return None
            
        # Train the model if not already trained
        if not self.model_trained:
            # Extract features and prepare data for training
            features = self.predictor._extract_features(data)
            self.predictor.scaler.fit(features)
            X, y = self.predictor._create_sequences(features)
            self.predictor.train(X, y)
            self.model_trained = True
            
            # Store the feature columns for prediction
            self.predictor._feature_columns = features.columns.tolist()
            
        results = {
            'dates': [],
            'actual_prices': [],
            'predicted_prices': [],
            'positions': [],
            'returns': [],
            'capital': []
        }
        
        window_size = 50
        for i in range(window_size, len(data) - 1):
            current_date = data.index[i]
            window_data = data.iloc[i-window_size:i]
            
            prediction = self.predictor.predict_next_day(window_data)
            actual_next_price = data.iloc[i+1]['Close']
            current_price = data.iloc[i]['Close']
            
            # Simplified market regime and volatility calculation for initial testing
            market_regime = 'normal'  # Default to normal regime for initial testing
            volatility = window_data['Close'].pct_change().std() * np.sqrt(252)
            
            if prediction > current_price * 1.01:  # Buy signal
                if self.positions == 0:
                    shares, metrics = self.risk_manager.calculate_position_size(
                        current_price, volatility, market_regime
                    )
                    cost = shares * current_price
                    if cost <= self.capital:
                        self.positions = shares
                        self.capital -= cost
                        self.trades.append({
                            'date': current_date,
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'cost': cost
                        })
            
            elif prediction < current_price * 0.99:  # Sell signal
                if self.positions > 0:
                    proceeds = self.positions * current_price
                    self.capital += proceeds
                    self.trades.append({
                        'date': current_date,
                        'type': 'sell',
                        'price': current_price,
                        'shares': self.positions,
                        'proceeds': proceeds
                    })
                    self.positions = 0
            
            portfolio_value = self.capital + (self.positions * current_price)
            daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            results['dates'].append(current_date)
            results['actual_prices'].append(actual_next_price)
            results['predicted_prices'].append(prediction)
            results['positions'].append(self.positions)
            results['returns'].append(daily_return)
            results['capital'].append(portfolio_value)
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results):
        if results is None or len(results) == 0:
            return None
            
        returns = pd.Series(results['returns'])
        metrics = {
            'total_return': ((results['capital'].iloc[-1] - self.initial_capital) 
                           / self.initial_capital * 100),
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() 
                          if returns.std() != 0 else 0,
            'max_drawdown': (returns.cummax() - returns).max() * 100,
            'win_rate': len([t for t in self.trades if t.get('proceeds', 0) > t.get('cost', 0)]) 
                       / len(self.trades) if len(self.trades) > 0 else 0,
            'total_trades': len(self.trades),
            'prediction_accuracy': np.mean(
                np.sign(np.diff(results['predicted_prices'])) == 
                np.sign(np.diff(results['actual_prices']))
            ) * 100
        }
        return metrics

def run_full_backtest():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    runner = BacktestRunner()
    results = runner.run_backtest(start_date, end_date)
    
    if results is not None:
        metrics = runner.calculate_metrics(results)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        plt.plot(results['dates'], results['actual_prices'], label='Actual Price', color='blue')
        plt.plot(results['dates'], results['predicted_prices'], label='Predicted Price', color='red', linestyle='--')
        plt.title(f'{SYMBOL} Price Prediction vs Actual')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(results['dates'], results['capital'], label='Portfolio Value', color='green')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
        
        return results, metrics, 'backtest_results.png'
    return None, None, None

if __name__ == '__main__':
    results, metrics, plot_path = run_full_backtest()
    if metrics:
        print(f"\nBacktest Results for {SYMBOL}:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
        print("\nPlot saved as backtest_results.png")
