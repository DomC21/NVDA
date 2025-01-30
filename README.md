# NVDA Trading Algorithm

A sophisticated trading algorithm focused on analyzing NVIDIA (NVDA) stock using multiple data sources and technical indicators.

## Technical Analysis Components

### 1. Moving Averages (20/50-day SMA)
- **20-day SMA**: Short-term trend indicator
  - Above price: Potential resistance level
  - Below price: Potential support level
- **50-day SMA**: Medium-term trend indicator
  - Crossovers with 20-day SMA signal trend changes
  - Golden Cross (20 above 50): Bullish
  - Death Cross (20 below 50): Bearish

### 2. Relative Strength Index (RSI)
- Momentum oscillator measuring speed/magnitude of price changes
- Scale: 0-100
- Key levels:
  - Above 70: Overbought condition, potential reversal
  - Below 30: Oversold condition, potential buying opportunity
  - 50 level: Trend strength confirmation

### 3. MACD (Moving Average Convergence Divergence)
- Trend-following momentum indicator
- Components:
  - MACD Line: 12-day EMA minus 26-day EMA
  - Signal Line: 9-day EMA of MACD
  - Histogram: MACD minus Signal Line
- Signals:
  - MACD crossing above Signal: Bullish
  - MACD crossing below Signal: Bearish
  - Histogram increasing: Strong momentum
  - Histogram decreasing: Weakening momentum

### 4. Bollinger Bands
- Volatility-based indicator (20-day SMA ± 2 standard deviations)
- Interpretation:
  - Price above upper band: Potentially overbought
  - Price below lower band: Potentially oversold
  - Band contraction: Low volatility, potential breakout
  - Band expansion: High volatility, trend strength

### 5. Volume Analysis
- Confirms price movements and trend strength
- Key aspects:
  - Volume increasing with price: Strong trend
  - Volume decreasing with price: Weak trend
  - Above average volume: Strong price action
  - Below average volume: Weak price action

### 6. Advanced Technical Indicators

#### Average True Range (ATR)
- Volatility indicator measuring market volatility
- Higher values indicate higher volatility
- Used for:
  - Stop loss placement
  - Position sizing
  - Volatility breakout strategies

#### Stochastic Oscillator
- Momentum indicator comparing closing price to price range
- Components:
  - %K (Fast Stochastic): Current price position relative to range
  - %D (Slow Stochastic): 3-period moving average of %K
- Signals:
  - Above 80: Overbought
  - Below 20: Oversold
  - Crossovers indicate potential reversals

#### On-Balance Volume (OBV)
- Volume-based momentum indicator
- Cumulative total of volume
- Interpretation:
  - Rising OBV with rising price: Strong uptrend
  - Falling OBV with rising price: Weak uptrend
  - Divergences signal potential reversals

#### Chaikin Money Flow (CMF)
- Volume-weighted accumulation/distribution
- 20-period default timeframe
- Signals:
  - Above 0: Buying pressure
  - Below 0: Selling pressure
  - Trend strength confirmation

#### Average Directional Index (ADX)
- Trend strength indicator
- Scale: 0-100
- Interpretation:
  - Above 25: Strong trend
  - Below 20: Weak trend
  - Not direction specific

### 7. Derivative Features

#### Price-Based
- Price Range: Daily high-low spread
- Gap Analysis: Opening price vs previous close
- Price Position: Relative position within daily range
- Rolling Volatility: 20-day standard deviation

#### Volume-Based
- Volume Intensity: Volume weighted by price movement
- Volume Force: Directional volume pressure
- Volatility Ratio: Current vs historical volatility

## Options Flow Analysis

### 1. Call/Put Ratio Analysis
- Measures institutional sentiment
- Interpretation:
  - Ratio > 0.6: Bullish sentiment (more calls)
  - Ratio < 0.4: Bearish sentiment (more puts)
  - 0.4-0.6: Neutral sentiment

### 2. Premium Analysis
- Dollar-weighted sentiment indicator
- Signals:
  - Premium Ratio > 0.6: Strong bullish conviction
  - Premium Ratio < 0.4: Strong bearish conviction
  - 0.4-0.6: Mixed institutional sentiment

## Signal Generation System

### Weighted Scoring Components
1. Technical Analysis (50% weight)
   - Trend Analysis: 40%
   - RSI: 20%
   - MACD: 20%
   - Volume: 10%
   - Bollinger Bands: 10%

2. Options Flow (30% weight)
   - Call/Put Ratio: 50%
   - Premium Analysis: 50%

3. Machine Learning Price Prediction (20% weight)
   - LSTM Neural Network Model
   - 60-day sequence analysis
   - Multi-feature input:
     - Historical prices
     - Volume trends
     - Technical indicators
   - Confidence ranges:
     - 90% prediction interval
     - 70% prediction interval
     - 50% prediction interval

### Price Prediction Model

The algorithm implements an ensemble approach combining multiple models for robust price prediction:

1. LSTM Neural Network
   - Input: 60-day sequence of features
   - 3-layer LSTM (128, 64, 32 neurons)
   - Dropout layers (0.3) for regularization
   - Dense output layer for price prediction

2. GRU Neural Network
   - Similar to LSTM but with simpler gating mechanism
   - Better performance on shorter sequences
   - Faster training time
   - 3-layer architecture (128, 64, 32 neurons)

3. XGBoost Regressor
   - Gradient boosting on decision trees
   - Handles non-linear relationships
   - Feature importance ranking
   - Early stopping to prevent overfitting

4. Gradient Boosting Regressor
   - Scikit-learn implementation
   - Robust to outliers
   - Less prone to overfitting
   - Complementary to XGBoost

### Ensemble Weighting
- LSTM: 30% (Strong on sequential patterns)
- GRU: 30% (Efficient on recent data)
- XGBoost: 20% (Non-linear relationships)
- Gradient Boosting: 20% (Robustness)

## Model Training and Optimization

### Data Preparation
1. Historical Data Collection
   - 2 years of daily price data
   - Technical indicators computation
   - Feature normalization using MinMaxScaler
   - Sequence creation (60-day windows)

### Cross-Validation
1. Time Series Split
   - 5 folds with expanding window
   - Maintains temporal order
   - Prevents future data leakage

2. Performance Metrics
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)
   - Root Mean Square Error (RMSE)

### Hyperparameter Tuning
1. LSTM/GRU Networks
   - Units per layer: [64, 128, 256]
   - Dropout rates: [0.2, 0.3, 0.4]
   - Learning rates: [0.001, 0.0005, 0.0001]
   - Batch sizes: [32, 64, 128]

2. XGBoost
   - max_depth: [3, 5, 7]
   - learning_rate: [0.01, 0.05, 0.1]
   - n_estimators: [100, 200, 300]
   - subsample: [0.8, 0.9, 1.0]

3. Gradient Boosting
   - n_estimators: [100, 200, 300]
   - learning_rate: [0.01, 0.05, 0.1]
   - max_depth: [3, 5, 7]
   - min_samples_split: [2, 5, 10]

### Training Process
1. Neural Networks
   - Early stopping (patience=10)
   - Reduce learning rate on plateau
   - Gradient clipping
   - Batch normalization

2. Tree-based Models
   - Early stopping with validation set
   - Feature importance analysis
   - Cross-validation scores

### Model Evaluation
1. Backtesting
   - Walk-forward optimization
   - Out-of-sample testing
   - Performance comparison

2. Ensemble Integration
   - Weighted averaging
   - Confidence intervals
   - Prediction consensus

2. Feature Engineering
   - Close prices
   - Trading volume
   - RSI (Relative Strength Index)
   - Moving averages (20-day, 50-day)
   - MACD (Moving Average Convergence Divergence)

3. Prediction Output
   - Next-day price prediction
   - Confidence intervals (90%, 70%, 50%)
   - Price movement signals:
     - Strong Buy: >2% predicted increase
     - Buy: 1-2% predicted increase
     - Neutral: -1% to 1% predicted change
     - Sell: -1% to -2% predicted decrease
     - Strong Sell: <-2% predicted decrease

4. Model Performance Metrics
   - Mean Absolute Error (MAE)
   - Mean Absolute Percentage Error (MAPE)
   - Root Mean Square Error (RMSE)
   - Backtesting results on historical data

### Signal Confidence Levels
- High (>70%): Strong conviction
- Medium (30-70%): Mixed signals
- Low (<30%): Weak conviction

## API Integration
- Unusual Whales: Options flow data
- Polygon: Market data and technical indicators
- Alpha Vantage: Additional market insights
- YFinance: Price data and historical information

## Sentiment Analysis

### News Sentiment Scoring
- Scale: 0-100%
- Interpretation:
  - 0-30%: Overwhelmingly negative
    - Major lawsuits
    - Significant earnings misses
    - Regulatory challenges
    - Executive departures
  
  - 30-45%: Moderately negative
    - Minor earnings misses
    - Market share decline
    - Competitive pressures
  
  - 45-55%: Neutral
    - Mixed market signals
    - Industry-wide trends
    - Uncertain market conditions
  
  - 55-70%: Moderately positive
    - Meeting earnings expectations
    - Product launches
    - Partnership announcements
  
  - 70-100%: Overwhelmingly positive
    - Record earnings
    - Market share gains
    - Strategic acquisitions
    - Industry leadership

### Integration with Signal Generation
- News sentiment contributes to the final signal through:
  - Short-term impact assessment
  - Trend confirmation/contradiction
  - Volume correlation analysis
  - Options flow validation
