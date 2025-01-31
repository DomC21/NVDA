import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analysis import StockAnalyzer
from price_predictor import PricePredictor
from config import UNUSUAL_WHALES_API_KEY, SYMBOL
from risk_manager import RiskManager
from drawdown_manager import DrawdownManager
from portfolio_risk_manager import PortfolioRiskManager
from volume_analyzer import VolumeAnalyzer
from market_regime_analyzer import MarketRegimeAnalyzer
from unusual_whales_analyzer import UnusualWhalesAnalyzer
from data_fetcher import DataFetcher

class TradingAlgorithm:
    """Trading algorithm for NVDA stock analysis combining technical indicators and options flow data.
    
    Technical Indicators:
    - Moving Averages (20/50-day): Trend identification and support/resistance levels
    - RSI (14-day): Momentum indicator identifying overbought (>70) or oversold (<30) conditions
    - MACD: Trend and momentum indicator using (12,26,9) day EMAs
    - Bollinger Bands: Volatility-based indicator using 20-day SMA ±2 standard deviations
    - Volume Analysis: Confirms price movements with trading volume strength
    """
    
    def __init__(self, portfolio_value: float = 100000.0):
        self.analyzer = StockAnalyzer()
        self.analyzer.prepare_data()
        self.predictor = PricePredictor()
        self.risk_manager = RiskManager(portfolio_value)
        self.drawdown_manager = DrawdownManager(portfolio_value)
        self.portfolio_risk_manager = PortfolioRiskManager(portfolio_value)
        self.data_fetcher = DataFetcher()
        self.unusual_whales_analyzer = UnusualWhalesAnalyzer()
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        self.technical_weight = 0.4
        self.options_weight = 0.2
        self.dark_pool_weight = 0.2
        self.prediction_weight = 0.2
        self.technical_weight = 0.5
        self.options_weight = 0.3
        self.prediction_weight = 0.2
        self.current_portfolio_value = portfolio_value
        
    def _get_market_data(self):
        """Fetches comprehensive market data from various sources."""
        try:
            dark_pool_data = self.data_fetcher.fetch_dark_pool_data(SYMBOL)
            option_flow_data = self.data_fetcher.fetch_option_flow(SYMBOL)
            market_tide_data = self.data_fetcher.fetch_market_tide()
            greeks_data = self.data_fetcher.fetch_greeks_exposure(SYMBOL)
            option_volume_data = self.data_fetcher.fetch_option_volume_levels(SYMBOL)
            
            return {
                'dark_pool': dark_pool_data,
                'option_flow': option_flow_data,
                'market_tide': market_tide_data,
                'greeks': greeks_data,
                'option_volume': option_volume_data
            }
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return {}
            
    def generate_trading_signals(self):
        """Generates comprehensive trading signals with risk management for NVDA stock."""
        technical_signals = self._generate_technical_signals()
        market_signals = self._analyze_market_data()
        prediction_signals = self._generate_prediction_signals()
        
        # Get market data for regime analysis
        market_data = self._get_market_data()
        
        # Get market regime analysis
        regime_analysis = self.market_regime_analyzer.analyze(
            self.analyzer.data,
            market_data.get('market_tide'),
            market_data.get('option_flow')
        )
        
        # Adjust signal weights based on market regime
        if regime_analysis.regime == 'trending':
            self.technical_weight = 0.5
            self.options_weight = 0.2
            self.dark_pool_weight = 0.2
            self.prediction_weight = 0.1
        elif regime_analysis.regime == 'volatile':
            self.technical_weight = 0.3
            self.options_weight = 0.3
            self.dark_pool_weight = 0.3
            self.prediction_weight = 0.1
        else:  # ranging or unknown
            self.technical_weight = 0.4
            self.options_weight = 0.2
            self.dark_pool_weight = 0.2
            self.prediction_weight = 0.2
            
        combined_signals = self._combine_signals(technical_signals, market_signals, prediction_signals)
        
        # Add risk management metrics
        current_price = self.analyzer.data['Close'].iloc[-1]
        volatility = self.analyzer.data['Close'].pct_change().std() * 100
        market_regime = 'trending' if technical_signals['confidence'] > 60 else 'ranging'
        
        shares, position_metrics = self.risk_manager.calculate_position_size(
            current_price, volatility, market_regime
        )
        
        atr = self.risk_manager.calculate_atr(
            self.analyzer.data['High'].values,
            self.analyzer.data['Low'].values,
            self.analyzer.data['Close'].values
        )
        
        stop_loss, profit_target = self.risk_manager.calculate_stop_loss(
            current_price, 
            'long' if combined_signals['signal'] == 'BUY' else 'short',
            atr
        )
        
        is_valid, validation_message = self.risk_manager.validate_trade(
            current_price, stop_loss, profit_target,
            'long' if combined_signals['signal'] == 'BUY' else 'short'
        )
        
        if not is_valid:
            combined_signals['signal'] = 'NEUTRAL'
            combined_signals['confidence'] *= 0.5
            combined_signals['technical_reasons'].append(f"Risk Management Override: {validation_message}")
            
        # Add drawdown protection
        is_valid_dd, dd_message, dd_metrics = self.drawdown_manager.validate_trade(self.current_portfolio_value)
        if not is_valid_dd:
            combined_signals['signal'] = 'NEUTRAL'
            combined_signals['confidence'] *= 0.3
            combined_signals['technical_reasons'].append(f"Drawdown Protection: {dd_message}")
            position_metrics['position_value'] *= dd_metrics['position_adjustment']
            position_metrics['position_size_pct'] *= dd_metrics['position_adjustment']
            
        # Add portfolio-level risk control
        is_valid_portfolio, portfolio_message, portfolio_metrics = self.portfolio_risk_manager.validate_new_position(
            symbol='NVDA',
            value=position_metrics['position_value'],
            sector='Technology',
            beta=self.analyzer.get_beta(),
            correlation_data=self.analyzer.get_correlations()
        )
        
        if not is_valid_portfolio:
            combined_signals['signal'] = 'NEUTRAL'
            combined_signals['confidence'] *= 0.2
            combined_signals['technical_reasons'].append(f"Portfolio Risk Control: {portfolio_message}")
            position_metrics['position_value'] *= self.portfolio_risk_manager.get_position_size_adjustment(
                portfolio_metrics.get('portfolio_risk', 0.0)
            )
            position_metrics['position_size_pct'] = position_metrics['position_value'] / self.current_portfolio_value
            
        return {
            'technical_signals': technical_signals,
            'market_signals': market_signals,
            'prediction_signals': prediction_signals,
            'market_regime': regime_analysis,
            'combined_recommendation': combined_signals,
            'risk_metrics': {
                'position_size': shares,
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'risk_amount': position_metrics['risk_amount'],
                'position_value': position_metrics['position_value'],
                'position_size_pct': position_metrics['position_size_pct'],
                'validation_message': validation_message
            }
        }
        
    def _generate_technical_signals(self):
        """Analyzes technical indicators to generate trading signals."""
        data = self.analyzer.data
        if data is None or len(data) < 50:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'reasons': ['Insufficient data']}
            
        # Add volume profile analysis
        volume_analyzer = VolumeAnalyzer(data)
        volume_profile = volume_analyzer.calculate_volume_profile()
        current_price = data['Close'].iloc[-1]
        volume_signals = volume_analyzer.get_entry_exit_signals(current_price)
        volume_trend = volume_analyzer.analyze_volume_trend()
        
        # Add market regime detection
        regime_detector = MarketRegimeDetector(data)
        regime_metrics = regime_detector.detect_regime()
        regime_params = regime_detector.get_regime_parameters()
        
        # Incorporate volume analysis and market regime into signals
        signals = []
        confidence_factors = []
        
        # Volume Profile Analysis (30% weight)
        volume_score = 0
        if volume_signals['long']['risk_reward_ratio'] > 2.0:
            volume_score += 50
            signals.append(f"Bullish: Strong risk/reward ratio ({volume_signals['long']['risk_reward_ratio']:.2f})")
        if volume_trend['volume_momentum'] > 0:
            volume_score += 50
            signals.append("Bullish: Positive volume momentum with price increase")
        elif volume_trend['volume_momentum'] < 0:
            signals.append("Bearish: Negative volume momentum with price decrease")
            
        confidence_factors.append(volume_score * 0.3)
        
        # Sentiment Analysis (10% weight)
        if self.sentiment_analyzer:
            news_items = self.sentiment_analyzer.fetch_news_sentiment('NVDA')
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(news_items)
            
            sentiment_score = (sentiment_analysis['composite_score'] + 1) * 50  # Convert -1,1 to 0,100
            confidence_factors.append(sentiment_score * 0.1)
            
            if sentiment_analysis['composite_score'] > 0.3:
                signals.append(f"Bullish: Strong positive sentiment (score: {sentiment_analysis['composite_score']:.2f})")
            elif sentiment_analysis['composite_score'] < -0.3:
                signals.append(f"Bearish: Strong negative sentiment (score: {sentiment_analysis['composite_score']:.2f})")
            
            if abs(sentiment_analysis['sentiment_momentum']) > 0.2:
                signals.append(f"{'Increasing' if sentiment_analysis['sentiment_momentum'] > 0 else 'Decreasing'} sentiment momentum")
        
        # Dark Pool Analysis (10% weight)
        if self.dark_pool_analyzer:
            dark_pool_trades = self.dark_pool_analyzer.fetch_dark_pool_data('NVDA')
            dark_pool_analysis = self.dark_pool_analyzer.analyze_dark_pool_activity(dark_pool_trades)
            significant_levels = self.dark_pool_analyzer.get_significant_levels(dark_pool_trades)
            
            dark_pool_score = (dark_pool_analysis['composite_score'] + 1) * 50  # Convert -1,1 to 0,100
            confidence_factors.append(dark_pool_score * 0.1)
            
            if dark_pool_analysis['composite_score'] > 0.3:
                signals.append(f"Bullish: Strong dark pool buying (score: {dark_pool_analysis['composite_score']:.2f})")
                if dark_pool_analysis['large_trade_impact'] > 0.4:
                    signals.append("Bullish: Significant large block trades detected")
            elif dark_pool_analysis['composite_score'] < -0.3:
                signals.append(f"Bearish: Strong dark pool selling (score: {dark_pool_analysis['composite_score']:.2f})")
                if dark_pool_analysis['large_trade_impact'] > 0.4:
                    signals.append("Bearish: Significant large block trades detected")
            
            if significant_levels:
                signals.append(f"Key dark pool levels: {', '.join([f'${level['price']:.2f}' for level in significant_levels[:3]])}")
        
        # Market Regime Analysis (20% weight)
        regime_score = 0
        if regime_metrics.regime.value == 'trending_up':
            regime_score = 100
            signals.append(f"Bullish: Strong upward trend detected (confidence: {regime_metrics.confidence:.2f})")
        elif regime_metrics.regime.value == 'trending_down':
            regime_score = 0
            signals.append(f"Bearish: Strong downward trend detected (confidence: {regime_metrics.confidence:.2f})")
        else:
            regime_score = 50
            signals.append(f"Neutral: {regime_metrics.regime.value} market detected")
            
        confidence_factors.append(regime_score * 0.2)
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]  # Previous day's data
        signals = []
        confidence_factors = []
        
        # Trend Analysis (40% weight)
        trend_score = 0
        if latest['Close'] > latest['SMA_20']:
            trend_score += 20
            signals.append('Bullish: Price above 20-day SMA indicating short-term uptrend')
        if latest['Close'] > latest['SMA_50']:
            trend_score += 20
            signals.append('Bullish: Price above 50-day SMA indicating medium-term uptrend')
        if latest['SMA_20'] > latest['SMA_50']:
            trend_score += 10
            signals.append('Bullish: Golden Cross pattern forming (20-day above 50-day)')
        confidence_factors.append(trend_score * 0.4)
        
        # RSI Analysis (20% weight)
        rsi_score = 0
        if latest['RSI'] < 30:
            rsi_score = 100
            signals.append('Bullish: RSI below 30 indicates oversold conditions')
        elif latest['RSI'] > 70:
            rsi_score = 0
            signals.append('Bearish: RSI above 70 indicates overbought conditions')
        else:
            rsi_score = 50
            if latest['RSI'] > prev['RSI']:
                signals.append('Neutral-Bullish: RSI showing increasing momentum')
            else:
                signals.append('Neutral-Bearish: RSI showing decreasing momentum')
        confidence_factors.append(rsi_score * 0.2)
        
        # MACD Analysis (20% weight)
        macd_score = 0
        if latest['MACD'] > latest['Signal_Line']:
            macd_score = 100
            signals.append('Bullish: MACD above signal line indicating upward momentum')
            if latest['MACD_Histogram'] > 0 and latest['MACD_Histogram'] > prev['MACD_Histogram']:
                signals.append('Bullish: MACD histogram increasing, strong momentum')
        else:
            signals.append('Bearish: MACD below signal line indicating downward momentum')
            if latest['MACD_Histogram'] < 0 and latest['MACD_Histogram'] < prev['MACD_Histogram']:
                signals.append('Bearish: MACD histogram decreasing, strong downward momentum')
        confidence_factors.append(macd_score * 0.2)
        
        # Volume Analysis (10% weight)
        volume_score = 0
        if latest['Volume'] > latest['Volume_SMA_20']:
            volume_score = 100
            if latest['Close'] > prev['Close']:
                signals.append('Bullish: Above average volume on price increase')
            else:
                signals.append('Bearish: Above average volume on price decrease')
        confidence_factors.append(volume_score * 0.1)
        
        # Bollinger Bands Analysis (10% weight)
        bb_score = 0
        if latest['Close'] > latest['BB_upper']:
            bb_score = 0
            signals.append('Bearish: Price above upper Bollinger Band indicating overbought')
        elif latest['Close'] < latest['BB_lower']:
            bb_score = 100
            signals.append('Bullish: Price below lower Bollinger Band indicating oversold')
        else:
            bb_score = 50
            if latest['Close'] > latest['BB_middle']:
                signals.append('Neutral-Bullish: Price above middle Bollinger Band')
            else:
                signals.append('Neutral-Bearish: Price below middle Bollinger Band')
        confidence_factors.append(bb_score * 0.1)
        
        total_confidence = sum(confidence_factors)
        signal = 'NEUTRAL'
        if total_confidence >= 70:
            signal = 'BUY'
        elif total_confidence <= 30:
            signal = 'SELL'
            
        return {
            'signal': signal,
            'confidence': total_confidence,
            'reasons': signals
        }
        
    def _analyze_market_data(self):
        """Analyzes comprehensive market data from multiple sources."""
        market_data = self._get_market_data()
        if not market_data:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'reasons': ['No market data available']}
            
        # Analyze dark pool activity
        dark_pool_analysis = self.unusual_whales_analyzer.analyze_dark_pool(market_data['dark_pool'])
        
        # Analyze options flow
        options_analysis = self.unusual_whales_analyzer.analyze_option_flow(market_data['option_flow'])
        
        # Analyze market tide
        market_tide_score = self.unusual_whales_analyzer.analyze_market_tide(market_data['market_tide'])
        
        # Analyze greeks exposure
        greek_exposure = self.unusual_whales_analyzer.analyze_greeks(market_data['greeks'])
        
        # Analyze volume levels
        volume_levels = self.unusual_whales_analyzer.analyze_volume_levels(market_data['option_volume'])
        
        # Generate comprehensive analysis
        analysis = UnusualWhalesAnalysis(
            dark_pool_metrics=dark_pool_analysis,
            option_flow_metrics=options_analysis,
            market_tide_score=market_tide_score,
            greek_exposure=greek_exposure,
            volume_levels=volume_levels,
            signal_strength=0.0
        )
        
        # Generate signal
        signal, confidence = self.unusual_whales_analyzer.generate_signal(analysis)
        
        signals = []
        if signal == 'buy':
            signals.extend([
                f"Bullish: Dark pool net flow {dark_pool_analysis['net_flow']:.2f}",
                f"Bullish: Options flow score {options_analysis['bullish_flow_score']:.2f}",
                f"Market tide score: {market_tide_score:.2f}",
                f"Net delta exposure: {greek_exposure['net_delta']:.2f}"
            ])
        else:
            signals.extend([
                f"Bearish: Dark pool net flow {dark_pool_analysis['net_flow']:.2f}",
                f"Bearish: Options flow score {options_analysis['bullish_flow_score']:.2f}",
                f"Market tide score: {market_tide_score:.2f}",
                f"Net delta exposure: {greek_exposure['net_delta']:.2f}"
            ])
            
        return {
            'signal': signal,
            'confidence': confidence,
            'reasons': signals
        }
        
    def _generate_prediction_signals(self):
        """Generates ML-based price prediction signals."""
        features = self.analyzer.data
        if features is None or len(features) < 60:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'reasons': ['Insufficient data for prediction']}
            
        try:
            self.predictor.build_model(input_shape=(60, features.shape[1]))
            next_day_price = self.predictor.predict_next_day(features)
            current_price = features['Close'].iloc[-1]
            price_ranges = self.predictor.generate_price_ranges(current_price, next_day_price)
            
            price_change_pct = ((next_day_price - current_price) / current_price) * 100
            
            signal = 'NEUTRAL'
            confidence = 50
            reasons = []
            
            if price_change_pct > 2:
                signal = 'BUY'
                confidence = min(50 + abs(price_change_pct) * 5, 100)
                reasons.append(f'Bullish: ML predicts {price_change_pct:.1f}% price increase')
            elif price_change_pct < -2:
                signal = 'SELL'
                confidence = min(50 + abs(price_change_pct) * 5, 100)
                reasons.append(f'Bearish: ML predicts {price_change_pct:.1f}% price decrease')
            else:
                reasons.append(f'Neutral: ML predicts minor price change of {price_change_pct:.1f}%')
                
            reasons.append(f'Current Price: ${current_price:.2f}')
            reasons.append(f'Predicted Price: ${next_day_price:.2f}')
            reasons.append(f'90% Confidence Range: ${price_ranges["confidence_ranges"]["90%"][0]:.2f} - ${price_ranges["confidence_ranges"]["90%"][1]:.2f}')
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasons': reasons,
                'price_prediction': price_ranges
            }
        except Exception as e:
            return {'signal': 'NEUTRAL', 'confidence': 50, 'reasons': [f'Error generating prediction: {str(e)}']}
            
    def _combine_signals(self, technical_signals, options_signals, prediction_signals):
        """Combines technical, options, and ML prediction signals into a final recommendation."""
        current_time = datetime.now()
        timing_recommendation = self.timing_optimizer.get_timing_recommendation(
            current_time, 
            'entry' if technical_signals['signal'] == 'BUY' else 'exit'
        )
        # Get market regime analysis
        regime_analysis = self.market_regime_analyzer.analyze(
            self.analyzer.data,
            market_data.get('market_tide'),
            market_data.get('option_flow')
        )
        
        # Adjust weights based on market regime
        if regime_analysis.regime == 'trending':
            self.technical_weight = 0.5
            self.options_weight = 0.2
            self.dark_pool_weight = 0.2
            self.prediction_weight = 0.1
        elif regime_analysis.regime == 'volatile':
            self.technical_weight = 0.3
            self.options_weight = 0.3
            self.dark_pool_weight = 0.3
            self.prediction_weight = 0.1
        
        combined_confidence = (
            technical_signals['confidence'] * self.technical_weight +
            options_signals['confidence'] * self.options_weight +
            dark_pool_analysis['net_flow'] * 100 * self.dark_pool_weight +
            prediction_signals['confidence'] * self.prediction_weight
        )
        
        signal = 'NEUTRAL'
        if combined_confidence >= 70:
            signal = 'BUY'
        elif combined_confidence <= 30:
            signal = 'SELL'
            
        # Calculate transaction costs and slippage
        current_price = self.analyzer.data['Close'].iloc[-1]
        avg_volume = self.analyzer.data['Volume'].rolling(window=20).mean().iloc[-1]
        volatility = self.analyzer.data['Close'].pct_change().rolling(window=20).std().iloc[-1]
        
        market_data = {
            'price': current_price,
            'avg_volume': avg_volume,
            'volatility': volatility,
            'side': signal.lower()
        }
        
        position_size = self.current_portfolio_value * 0.1  # 10% position size
        transaction_costs = self.slippage_model.estimate_transaction_cost(
            price=current_price,
            size=position_size,
            market_data=market_data
        )
        
        # Optimize order size based on slippage constraints
        max_slippage = 0.003  # 0.3% max slippage
        optimal_size = self.slippage_model.optimize_order_size(
            target_size=position_size,
            max_slippage=max_slippage,
            market_data=market_data
        )
        
        # Adjust confidence based on transaction costs
        cost_ratio = transaction_costs['total_cost'] / (current_price * optimal_size)
        if cost_ratio > 0.005:  # If costs > 0.5% of position value
            combined_confidence *= 0.8  # Reduce confidence by 20%
            
        # Update signal if costs are too high
        if cost_ratio > 0.01:  # If costs > 1% of position value
            signal = 'NEUTRAL'
            combined_confidence *= 0.5
            
        return {
            'signal': signal,
            'confidence': combined_confidence,
            'technical_reasons': technical_signals['reasons'],
            'options_reasons': options_signals['reasons'],
            'prediction_reasons': prediction_signals['reasons'],
            'price_prediction': prediction_signals.get('price_prediction', {}),
            'transaction_analysis': {
                'estimated_costs': transaction_costs,
                'optimal_size': optimal_size,
                'cost_ratio': cost_ratio,
                'max_slippage': max_slippage,
                'timing': timing_recommendation
            }
        }
