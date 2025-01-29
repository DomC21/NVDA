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
1. Technical Analysis (60% weight)
   - Trend Analysis: 40%
   - RSI: 20%
   - MACD: 20%
   - Volume: 10%
   - Bollinger Bands: 10%

2. Options Flow (40% weight)
   - Call/Put Ratio: 50%
   - Premium Analysis: 50%

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
