from price_predictor import PricePredictor

def main():
    predictor = PricePredictor()
    
    print("\nRunning NVDA Price Prediction Backtest...")
    backtest_results = predictor.backtest()
    
    print("\nModel Performance Metrics:")
    print("=" * 50)
    print(f"Mean Absolute Error: ${backtest_results['mae']:.2f}")
    print(f"Mean Absolute Percentage Error: {backtest_results['mape']:.2f}%")
    print(f"Root Mean Square Error: ${backtest_results['rmse']:.2f}")
    
    print("\nTraining Performance:")
    print("-" * 30)
    final_loss = backtest_results['training_history']['loss'][-1]
    final_mae = backtest_results['training_history']['mae'][-1]
    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Final Training MAE: {final_mae:.4f}")
    
    features = predictor._extract_features(predictor.collector.collect_all_data()['yfinance'])
    next_day_price = predictor.predict_next_day(features)
    ranges = predictor.generate_price_ranges(features['Close'].iloc[-1], next_day_price)
    
    print("\nNext Day Price Prediction:")
    print("=" * 50)
    print(f"Current Price: ${features['Close'].iloc[-1]:.2f}")
    print(f"Predicted Next Day Price: ${ranges['prediction']:.2f}")
    print("\nConfidence Ranges:")
    for confidence, (lower, upper) in ranges['confidence_ranges'].items():
        print(f"{confidence} Range: ${lower:.2f} - ${upper:.2f}")

if __name__ == "__main__":
    main()
