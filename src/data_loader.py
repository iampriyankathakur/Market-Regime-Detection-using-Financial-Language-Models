import yfinance as yf
import pandas as pd

def load_market_data(symbol="SPY", start="2015-01-01"):
    df = yf.download(symbol, start=start)
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna()
    return df

def load_mock_news():
    # Replace with real macro news dataset later
    texts = [
        "Inflation pressures continue to rise amid supply constraints",
        "Federal Reserve signals possible rate cuts next quarter",
        "Economic growth remains stable despite volatility",
        "Recession fears increase as consumer spending slows",
        "Strong labor market supports bullish outlook"
    ] * 200
    return texts
