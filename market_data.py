from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # lets the app still run if yfinance is missing
    yf = None


def yf_symbol(symbol: str) -> str:
    """Convert broker symbols to Yahoo Finance style when possible."""
    s = str(symbol).strip().upper()
    return s.replace("/", "-")


def _empty_tech(symbol: str, reason: str = "No data") -> dict:
    return {
        "Symbol": symbol,
        "Last Close": np.nan,
        "EMA 10": np.nan,
        "EMA 20": np.nan,
        "EMA 50": np.nan,
        "MA 200": np.nan,
        "Trend Rating": "No data",
        "Trend Score": 0,
        "Trend Setup": reason,
        "Distance from 50 EMA %": np.nan,
        "Distance from 200 MA %": np.nan,
        "52W High": np.nan,
        "52W Low": np.nan,
        "Drawdown from 52W High %": np.nan,
    }


def fetch_technical_snapshot(symbols: Iterable[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Pull price history and calculate trend structure.

    Bullish = EMA10 > EMA20 > EMA50 and Close > MA200.
    Bearish = EMA10 < EMA20 < EMA50 and Close < MA200.

    Supported intervals: 1h, 4h, 1d, 1wk. 4h is built by resampling Yahoo hourly candles.
    """
    rows = []
    if yf is None:
        return pd.DataFrame([_empty_tech(s, "Install yfinance") for s in symbols])

    fetch_interval = "1h" if interval == "4h" else interval
    for sym in list(dict.fromkeys([str(s).upper() for s in symbols if str(s).strip()])):
        try:
            hist = yf.Ticker(yf_symbol(sym)).history(period=period, interval=fetch_interval, auto_adjust=False)
            if hist is None or hist.empty or "Close" not in hist:
                rows.append(_empty_tech(sym))
                continue
            hist = hist.dropna(subset=["Close"]).copy()
            if interval == "4h":
                # Yahoo does not have a clean 4h interval. Build it from hourly candles.
                hist = hist.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
            if len(hist) < 55:
                rows.append(_empty_tech(sym, "Not enough candles"))
                continue
            close = hist["Close"].astype(float)
            ema10 = close.ewm(span=10, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ma200 = close.rolling(window=200).mean() if len(close) >= 200 else pd.Series(np.nan, index=close.index)
            c = float(close.iloc[-1])
            e10, e20, e50 = float(ema10.iloc[-1]), float(ema20.iloc[-1]), float(ema50.iloc[-1])
            m200 = float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else np.nan

            bullish = (e10 > e20 > e50) and pd.notna(m200) and (c > m200)
            bearish = (e10 < e20 < e50) and pd.notna(m200) and (c < m200)
            if bullish:
                rating = "Bullish"
                score = 1
                setup = "EMA10 > EMA20 > EMA50 and Close > MA200"
            elif bearish:
                rating = "Bearish"
                score = -1
                setup = "EMA10 < EMA20 < EMA50 and Close < MA200"
            else:
                rating = "Neutral"
                score = 0
                setup = "Mixed EMA structure"

            high52 = float(close.tail(min(len(close), 252)).max())
            low52 = float(close.tail(min(len(close), 252)).min())
            rows.append({
                "Symbol": sym,
                "Timeframe": interval,
                "Last Close": c,
                "EMA 10": e10,
                "EMA 20": e20,
                "EMA 50": e50,
                "MA 200": m200,
                "Trend Rating": rating,
                "Trend Score": score,
                "Trend Setup": setup,
                "Distance from 50 EMA %": (c / e50 - 1) if e50 else np.nan,
                "Distance from 200 MA %": (c / m200 - 1) if pd.notna(m200) and m200 else np.nan,
                "52W High": high52,
                "52W Low": low52,
                "Drawdown from 52W High %": (c / high52 - 1) if high52 else np.nan,
            })
        except Exception as exc:
            rows.append(_empty_tech(sym, str(exc)[:80]))
    return pd.DataFrame(rows)


def fetch_fundamentals(symbols: Iterable[str]) -> pd.DataFrame:
    rows = []
    if yf is None:
        return pd.DataFrame({"Symbol": list(symbols), "Market Data Status": "Install yfinance"})
    for sym in list(dict.fromkeys([str(s).upper() for s in symbols if str(s).strip()])):
        row = {"Symbol": sym, "Market Data Status": "OK"}
        try:
            t = yf.Ticker(yf_symbol(sym))
            fast = {}
            info = {}
            try:
                fast = dict(t.fast_info or {})
            except Exception:
                fast = {}
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}

            row.update({
                "Name": info.get("shortName") or info.get("longName") or "",
                "Sector": info.get("sector") or "",
                "Industry": info.get("industry") or "",
                "Market Cap": info.get("marketCap") or fast.get("market_cap"),
                "Trailing P/E": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "Price/Sales": info.get("priceToSalesTrailing12Months"),
                "Price/Book": info.get("priceToBook"),
                "EV/EBITDA": info.get("enterpriseToEbitda"),
                "Dividend Yield": info.get("dividendYield"),
                "Beta": info.get("beta"),
                "Analyst Target Mean": info.get("targetMeanPrice"),
                "Recommendation": info.get("recommendationKey") or "",
                "52W High Live": fast.get("year_high"),
                "52W Low Live": fast.get("year_low"),
            })
        except Exception as exc:
            row["Market Data Status"] = str(exc)[:120]
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_returns(symbols: Iterable[str], period: str = "1y") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    data = {}
    for sym in list(dict.fromkeys([str(s).upper() for s in symbols if str(s).strip()])):
        try:
            h = yf.Ticker(yf_symbol(sym)).history(period=period, interval="1d", auto_adjust=True)
            if h is not None and not h.empty and "Close" in h:
                data[sym] = h["Close"].pct_change().dropna()
        except Exception:
            pass
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).dropna(how="all")


def max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    curve = (1 + returns.fillna(0)).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    return float(dd.min())
