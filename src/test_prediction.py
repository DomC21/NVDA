from price_predictor import PricePredictor

def main():
    predictor = PricePredictor()
    
    print("\nPreparing NVDA price prediction model...")
    X, y = predictor.prepare_data()
    
    print("\nBuilding and training model...")
    predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
    predictor.train(X, y, epochs=50)
    
    features = predictor._extract_features(predictor.collector.collect_all_data()['yfinance'])
    next_day_price = predictor.predict_next_day(features)
    
    ranges = predictor.generate_price_ranges(features['Close'].iloc[-1], next_day_price)
    
    print("\nNVDA Price Prediction Report:")
    print("=" * 50)
    print(f"Current Price: ${features['Close'].iloc[-1]:.2f}")
    print(f"Predicted Next Day Price: ${ranges['prediction']:.2f}")
    print("\nConfidence Ranges:")
    for confidence, (lower, upper) in ranges['confidence_ranges'].items():
        print(f"{confidence} Range: ${lower:.2f} - ${upper:.2f}")

if __name__ == "__main__":
    main()
