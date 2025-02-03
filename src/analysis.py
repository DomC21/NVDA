"""Stock analysis module for NVDA trading."""
from typing import Dict, Any, List
import pandas as pd
from data_collector import DataCollector


class StockAnalyzer:
    def __init__(self):
        self.collector = DataCollector()
        self.data = None

    def prepare_data(self) -> None:
        """Fetch and prepare data for analysis"""
        raw_data = self.collector.collect_all_data()

        # Use YFinance as primary source due to data quality and consistency
        if ('yfinance' not in raw_data or raw_data['yfinance'] is None
                or len(raw_data['yfinance']) == 0):
            raise ValueError("Failed to fetch YFinance data")

        self.data = raw_data['yfinance']

        # Calculate beta and correlations
        self._calculate_beta()
        self._calculate_correlations()

        # Calculate technical indicators
        self.calculate_indicators()

    def calculate_indicators(self) -> None:
        """Calculate technical indicators for NVDA"""
        if not self.data or len(self.data) == 0:
            msg = "No data available. Call prepare_data() first."
            raise ValueError(msg)

        # Get price and volume data
        try:
            cols = self.data.columns
            close_col = 'Close' if 'Close' in cols else 'close'
            volume_col = 'Volume' if 'Volume' in cols else 'volume'
            close_series = self.data[close_col]
            volume_series = self.data[volume_col]
        except KeyError as e:
            msg = f"Required price data not found: {e}"
            raise ValueError(msg)

        # Simple Moving Averages
        self.data['SMA_20'] = close_series.rolling(window=20).mean()
        self.data['SMA_50'] = close_series.rolling(window=50).mean()

        # Relative Strength Index (14-day)
        delta = close_series.diff()
        gain = (delta.where(delta.gt(0.0), 0.0)).rolling(window=14).mean()
        loss = (-delta.where(delta.lt(0.0), 0.0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12,26,9)
        macd_fast, macd_slow, macd_signal = 12, 26, 9
        exp1 = close_series.ewm(span=macd_fast, adjust=False).mean()
        exp2 = close_series.ewm(span=macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        self.data['MACD'] = macd
        signal = macd.ewm(span=macd_signal, adjust=False).mean()
        self.data['Signal_Line'] = signal
        self.data['MACD_Histogram'] = macd - signal

        # Bollinger Bands
        bb_window = 20
        bb_middle = close_series.rolling(window=bb_window).mean()
        bb_std = close_series.rolling(window=bb_window).std()
        self.data['BB_middle'] = bb_middle
        self.data['BB_upper'] = bb_middle + (bb_std * 2.0)
        self.data['BB_lower'] = bb_middle - (bb_std * 2.0)

        # Volume Analysis
        self.data['Volume_SMA_20'] = volume_series.rolling(window=20).mean()

    def get_current_signals(self) -> List[str]:
        """Generate trading signals based on technical indicators"""
        if self.data is None or len(self.data) < 50:
            return ["Insufficient data for analysis"]

        latest = self.data.iloc[-1]
        signals = []

        # Trend Analysis
        if latest['Close'] > latest['SMA_50']:
            signals.append("BULLISH: Above 50-day SMA, uptrend")
        else:
            signals.append("BEARISH: Below 50-day SMA, downtrend")

        # RSI Analysis
        if latest['RSI'] > 70:
            signals.append("OVERBOUGHT: RSI > 70, reversal likely")
        elif latest['RSI'] < 30:
            signals.append("OVERSOLD: RSI < 30, buying opportunity")

        # MACD Analysis
        if latest['MACD'] > latest['Signal_Line']:
            signals.append("BULLISH: MACD above signal line")
        else:
            signals.append("BEARISH: MACD below signal line")

        # Volume Analysis
        if latest['Volume'] > latest['Volume_SMA_20']:
            signals.append("HIGH VOLUME: Above average")
        return signals

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for NVDA"""
        if self.data is None:
            return {"error": "No data available"}

        daily_returns = self.data['Close'].pct_change()
        return {
            'current_price': float(self.data['Close'].iloc[-1]),
            'daily_return': float(daily_returns.iloc[-1] * 100),
            'volatility': float(daily_returns.std() * 100),
            'avg_volume': float(self.data['Volume'].mean()),
            'rsi': float(self.data['RSI'].iloc[-1]),
            'macd': float(self.data['MACD'].iloc[-1]),
            'beta': self.get_beta(),
            'correlations': self.get_correlations()
        }

    def _calculate_beta(self) -> None:
        """Calculate beta using S&P 500 as market proxy"""
        try:
            spy_data = self.collector.collect_market_data('SPY')
            if (spy_data is None or not isinstance(spy_data, pd.DataFrame) or
                    len(spy_data) < 30):
                self.beta = 1.0
                return

            if self.data is None or 'Close' not in self.data.columns:
                self.beta = 1.0
                return

            stock_returns = self.data['Close'].pct_change().dropna()
            market_returns = spy_data['Close'].pct_change().dropna()

            aligned_data = pd.concat(
                [stock_returns, market_returns], axis=1
            ).dropna()
            if len(aligned_data) < 30:
                self.beta = 1.0
                return

            covariance = float(
                aligned_data.iloc[:, 0].cov(aligned_data.iloc[:, 1])
            )
            market_variance = float(aligned_data.iloc[:, 1].var())
            self.beta = (
                covariance / market_variance if market_variance != 0.0 else 1.0
            )
        except Exception as e:
            print(f"Error calculating beta: {e}")
            self.beta = 1.0

    def _calculate_correlations(self) -> None:
        """Calculate correlations with other tech stocks"""
        try:
            tech_stocks = ['AAPL', 'AMD', 'INTC', 'TSM']
            correlations: Dict[str, float] = {}

            if self.data is None or 'Close' not in self.data.columns:
                self.correlations = {stock: 0.0 for stock in tech_stocks}
                return

            for stock in tech_stocks:
                stock_data = self.collector.collect_market_data(stock)
                valid_data = (
                    stock_data is not None and
                    isinstance(stock_data, pd.DataFrame) and
                    len(stock_data) > 30
                )
                if valid_data:
                    if 'Close' not in stock_data.columns:
                        correlations[stock] = 0.0
                        continue

                    stock_returns = stock_data['Close'].pct_change().dropna()
                    nvda_returns = self.data['Close'].pct_change().dropna()

                    data_pair = [nvda_returns, stock_returns]
                    aligned_data = pd.concat(data_pair, axis=1).dropna()

                    if len(aligned_data) > 30:
                        corr = aligned_data.iloc[:, 0].corr(
                            aligned_data.iloc[:, 1]
                        )
                        correlation = float(corr)
                        correlations[stock] = correlation
                    else:
                        correlations[stock] = 0.0
                else:
                    correlations[stock] = 0.0

            self.correlations = correlations
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            self.correlations = {stock: 0.0 for stock in tech_stocks}

    def get_beta(self) -> float:
        """Returns the stock's beta coefficient"""
        has_valid_beta = hasattr(self, 'beta') and self.beta is not None
        return self.beta if has_valid_beta else 1.0

    def get_correlations(self) -> dict:
        """Returns correlations with other tech stocks"""
        has_valid_corr = hasattr(self, 'correlations') and self.correlations
        return self.correlations if has_valid_corr else {}
