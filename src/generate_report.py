from price_predictor import PricePredictor
from trading_algorithm import TradingAlgorithm
from data_collector import DataCollector
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_analysis_report():
    # Initialize components
    predictor = PricePredictor()
    algo = TradingAlgorithm()
    collector = DataCollector()

    # Get current data and predictions
    data = collector.collect_all_data()
    features = predictor._extract_features(data['yfinance'])
    next_day_price = predictor.predict_next_day(features)
    ranges = predictor.generate_price_ranges(features['Close'].iloc[-1], next_day_price)
    signals = algo.generate_trading_signals()

    # Create price prediction plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot historical prices
    plt.plot(features.index[-30:], features['Close'][-30:], label='Historical Price', color='blue')
    
    # Plot prediction point
    last_date = features.index[-1]
    next_date = pd.Timedelta(days=1) + last_date
    plt.scatter(next_date, ranges['prediction'], color='green', s=100, label='Prediction')
    
    # Plot confidence intervals
    for confidence, (lower, upper) in ranges['confidence_ranges'].items():
        alpha = float(confidence.strip('%')) / 100
        plt.fill_between([next_date], [lower], [upper], alpha=0.2, color='green', 
                        label=f'{confidence} Confidence')
    
    plt.title('NVDA Price Prediction with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nvda_prediction.png')
    plt.close()

    # Generate text report
    report = f"""
NVDA Stock Analysis and Price Prediction
{'=' * 50}

Current Price: ${features['Close'].iloc[-1]:.2f}
Predicted Next Day Price: ${ranges['prediction']:.2f}

Confidence Ranges:
{'-' * 30}"""
    
    for confidence, (lower, upper) in ranges['confidence_ranges'].items():
        report += f"\n{confidence} Range: ${lower:.2f} - ${upper:.2f}"
    
    report += f"""

Trading Signals:
{'-' * 30}
Signal: {signals['combined_recommendation']['signal']}
Confidence: {signals['combined_recommendation']['confidence']:.2f}%

Technical Analysis Reasons:
{'-' * 30}"""
    
    for reason in signals['combined_recommendation']['technical_reasons']:
        report += f"\n- {reason}"
    
    report += f"""

Options Flow Analysis:
{'-' * 30}"""
    
    for reason in signals['combined_recommendation']['options_reasons']:
        report += f"\n- {reason}"
    
    return report

if __name__ == "__main__":
    print(generate_analysis_report())
