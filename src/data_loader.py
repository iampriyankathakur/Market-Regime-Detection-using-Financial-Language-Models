import yfinance as yf
import pandas as pd
from datetime import datetime

def load_market_data(symbol, start_date):
    df = yf.download(symbol, start=start_date)
    df["Returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

def generate_macro_news(df):
    """
    Synthetic macro narratives aligned with dates.
    Replace with real API ingestion in production.
    """
    texts = []
    for date in df.index:
        if df.loc[date]["Returns"] > 0:
            texts.append("Economic outlook improving with strong labor and growth data")
        else:
            texts.append("Market uncertainty rising amid inflation and recession fears")
    return texts
