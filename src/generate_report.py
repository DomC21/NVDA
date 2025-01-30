from price_predictor import PricePredictor
from trading_algorithm import TradingAlgorithm
from data_collector import DataCollector
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _get_sentiment_label(score):
    if score >= 0.7:
        return "Very Bullish"
    elif score >= 0.6:
        return "Bullish"
    elif score >= 0.4:
        return "Neutral"
    elif score >= 0.3:
        return "Bearish"
    else:
        return "Very Bearish"

def generate_analysis_report():
    # Initialize components
    predictor = PricePredictor()
    algo = TradingAlgorithm()
    collector = DataCollector()

    # Get current data and predictions
    data = collector.collect_all_data()
    features = predictor._extract_features(data['yfinance'])
    
    # Train the model with cross-validation
    X, y = predictor._create_sequences(features)
    split_idx = int(len(X) * 0.8)  # 80% training data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Perform cross-validation with hyperparameter tuning
    param_grid = {
        'lstm_units': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dense_units': [16, 32, 64]
    }
    
    best_params, cv_results = predictor.grid_search_cv(X_train, y_train, param_grid, n_splits=5)
    predictor.model = predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]), params=best_params)
    history = predictor.train(X_train, y_train, epochs=50)
    
    # Generate predictions
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
    # Calculate sentiment metrics
    current_sentiment = features['News_Sentiment'].iloc[-1]
    sentiment_ma5 = features['Sentiment_MA5'].iloc[-1]
    sentiment_ma10 = features['Sentiment_MA10'].iloc[-1]
    
    report = f"""
NVDA Stock Analysis and Price Prediction
{'=' * 50}

Current Price: ${features['Close'].iloc[-1]:.2f}
Predicted Next Day Price: ${ranges['prediction']:.2f}

Market Sentiment:
{'-' * 30}
Current Sentiment: {current_sentiment:.2%} ({_get_sentiment_label(current_sentiment)})
5-Day Sentiment Trend: {sentiment_ma5:.2%}
10-Day Sentiment Trend: {sentiment_ma10:.2%}

Confidence Ranges:
{'-' * 30}"""
    
    for confidence, (lower, upper) in ranges['confidence_ranges'].items():
        report += f"\n{confidence} Range: ${lower:.2f} - ${upper:.2f}"
    
    # Calculate cross-validation metrics summary
    cv_mae_mean = np.mean([result['mean_mae'] for result in predictor.cv_results])
    cv_mae_std = np.mean([result['std_mae'] for result in predictor.cv_results])
    
    report += f"""

Model Performance (Cross-Validation):
{'-' * 30}
Mean Absolute Error: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}
Best Parameters:
- LSTM Units: {best_params['lstm_units']}
- Dropout Rate: {best_params['dropout_rate']}
- Learning Rate: {best_params['learning_rate']}
- Dense Units: {best_params['dense_units']}

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
