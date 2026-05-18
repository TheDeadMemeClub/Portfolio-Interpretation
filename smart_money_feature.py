
"""
smart_money_feature.py

Drop this file into the same folder as your Streamlit portfolio app.
Then in your main app file, add:

    from smart_money_feature import render_smart_money_tab

Inside your Streamlit tabs:
    tab1, tab2, tab3 = st.tabs(["Portfolio", "Technical Analysis", "Smart Money Planner"])
    with tab3:
        render_smart_money_tab()

This module adds a rule-based Smart Money / M-Block trade planner, stock analyzer,
risk calculator, and trade journal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


PERIOD_OPTIONS = {
    "1D": "1d",
    "5D": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "10Y": "10y",
    "MAX": "max",
}

INTERVAL_OPTIONS = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


@dataclass
class TradeInputs:
    ticker: str
    direction: str
    account_size: float
    risk_pct: float
    entry: float
    stop: float
    target: float


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance MultiIndex columns safely."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten_yf_columns(df).reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={date_col: "Date"})
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    df = df[needed].dropna(subset=["Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for span in [10, 20, 50, 200]:
        out[f"EMA{span}"] = out["Close"].ewm(span=span, adjust=False).mean()
    out["Range"] = out["High"] - out["Low"]
    out["Body"] = (out["Close"] - out["Open"]).abs()
    out["BodyPctRange"] = np.where(out["Range"] > 0, out["Body"] / out["Range"], 0)
    out["AvgRange20"] = out["Range"].rolling(20, min_periods=5).mean()
    out["AvgVol20"] = out["Volume"].rolling(20, min_periods=5).mean()
    return out


def detect_swings(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["SwingHigh"] = False
    out["SwingLow"] = False
    if len(out) < window * 2 + 1:
        return out

    highs = out["High"].values
    lows = out["Low"].values

    for i in range(window, len(out) - window):
        out.loc[out.index[i], "SwingHigh"] = highs[i] == np.max(highs[i - window : i + window + 1])
        out.loc[out.index[i], "SwingLow"] = lows[i] == np.min(lows[i - window : i + window + 1])
    return out


def classify_trend(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 60:
        return "Not enough data"

    last = df.iloc[-1]
    bullish_ema = last["EMA10"] > last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA200"]
    bearish_ema = last["EMA10"] < last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA200"]

    swing_highs = df[df["SwingHigh"]].tail(3)
    swing_lows = df[df["SwingLow"]].tail(3)

    higher_lows = len(swing_lows) >= 2 and swing_lows["Low"].iloc[-1] > swing_lows["Low"].iloc[-2]
    higher_highs = len(swing_highs) >= 2 and swing_highs["High"].iloc[-1] > swing_highs["High"].iloc[-2]
    lower_lows = len(swing_lows) >= 2 and swing_lows["Low"].iloc[-1] < swing_lows["Low"].iloc[-2]
    lower_highs = len(swing_highs) >= 2 and swing_highs["High"].iloc[-1] < swing_highs["High"].iloc[-2]

    if bullish_ema and (higher_lows or higher_highs):
        return "Bullish"
    if bearish_ema and (lower_lows or lower_highs):
        return "Bearish"
    if bullish_ema:
        return "Bullish EMA alignment, structure needs confirmation"
    if bearish_ema:
        return "Bearish EMA alignment, structure needs confirmation"
    return "Neutral / Choppy"


def detect_bos(df: pd.DataFrame) -> Tuple[str, Optional[float]]:
    """Detect simple body-close break of recent swing level."""
    if df.empty or len(df) < 30:
        return "No clear BOS", None

    last_close = float(df["Close"].iloc[-1])
    prev_swing_highs = df[df["SwingHigh"]].iloc[:-1].tail(3)
    prev_swing_lows = df[df["SwingLow"]].iloc[:-1].tail(3)

    if not prev_swing_highs.empty:
        recent_high = float(prev_swing_highs["High"].max())
        if last_close > recent_high:
            return "Bullish BOS / displacement body close", recent_high

    if not prev_swing_lows.empty:
        recent_low = float(prev_swing_lows["Low"].min())
        if last_close < recent_low:
            return "Bearish BOS / displacement body close", recent_low

    return "No fresh body-close BOS", None


def detect_market_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic market block detector:
    - Finds displacement candles with range > 1.25x average range
    - Body is at least 55% of range
    - Tags previous candle/cluster as a potential MB.
    This is intentionally conservative and should be manually confirmed.
    """
    out = df.copy()
    out["Displacement"] = (out["Range"] > 1.25 * out["AvgRange20"]) & (out["BodyPctRange"] > 0.55)
    out["BullishDisplacement"] = out["Displacement"] & (out["Close"] > out["Open"])
    out["BearishDisplacement"] = out["Displacement"] & (out["Close"] < out["Open"])
    out["PotentialDemandMB"] = False
    out["PotentialSupplyMB"] = False

    for i in range(2, len(out)):
        if out["BullishDisplacement"].iloc[i]:
            # Last down candle before displacement
            for j in range(i - 1, max(i - 6, -1), -1):
                if out["Close"].iloc[j] < out["Open"].iloc[j]:
                    out.loc[out.index[j], "PotentialDemandMB"] = True
                    break
        if out["BearishDisplacement"].iloc[i]:
            # Last up candle before displacement
            for j in range(i - 1, max(i - 6, -1), -1):
                if out["Close"].iloc[j] > out["Open"].iloc[j]:
                    out.loc[out.index[j], "PotentialSupplyMB"] = True
                    break
    return out


def recent_zones(df: pd.DataFrame, max_zones: int = 8) -> pd.DataFrame:
    zones = []
    for _, row in df[df["PotentialDemandMB"]].tail(max_zones).iterrows():
        zones.append({
            "Date": row["Date"],
            "Type": "Demand MB / possible SND",
            "Low": float(row["Low"]),
            "High": float(row["High"]),
            "50%": float((row["Low"] + row["High"]) / 2),
            "Status": "Manual confirm: valid until body close below zone",
        })
    for _, row in df[df["PotentialSupplyMB"]].tail(max_zones).iterrows():
        zones.append({
            "Date": row["Date"],
            "Type": "Supply MB / possible SND",
            "Low": float(row["Low"]),
            "High": float(row["High"]),
            "50%": float((row["Low"] + row["High"]) / 2),
            "Status": "Manual confirm: valid until body close above zone",
        })
    if not zones:
        return pd.DataFrame(columns=["Date", "Type", "Low", "High", "50%", "Status"])
    return pd.DataFrame(zones).sort_values("Date", ascending=False).head(max_zones)


def make_chart(df: pd.DataFrame, ticker: str, zones_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=ticker.upper(),
    ))

    for ema in ["EMA10", "EMA20", "EMA50", "EMA200"]:
        if ema in df.columns and df[ema].notna().any():
            fig.add_trace(go.Scatter(x=df["Date"], y=df[ema], name=ema, mode="lines"))

    # Mark swings
    highs = df[df["SwingHigh"]]
    lows = df[df["SwingLow"]]
    fig.add_trace(go.Scatter(x=highs["Date"], y=highs["High"], mode="markers", name="Swing High"))
    fig.add_trace(go.Scatter(x=lows["Date"], y=lows["Low"], mode="markers", name="Swing Low"))

    # Recent zones as rectangles
    if not zones_df.empty:
        x0 = df["Date"].iloc[max(0, len(df) - 120)]
        x1 = df["Date"].iloc[-1]
        for _, z in zones_df.head(5).iterrows():
            fig.add_hrect(
                y0=z["Low"],
                y1=z["High"],
                line_width=1,
                opacity=0.12,
                annotation_text=z["Type"],
                annotation_position="top left",
            )

    fig.update_layout(
        title=f"{ticker.upper()} Smart Money / M-Block Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def calc_risk(inp: TradeInputs) -> Dict[str, float | str]:
    entry, stop, target = inp.entry, inp.stop, inp.target
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or inp.account_size <= 0 or inp.risk_pct <= 0:
        return {"error": "Entry, stop, account size, and risk must be valid."}

    dollar_risk = inp.account_size * (inp.risk_pct / 100)
    shares = np.floor(dollar_risk / risk_per_share)

    if inp.direction == "Long":
        reward_per_share = target - entry
        one_r = entry + risk_per_share
        two_r = entry + 2 * risk_per_share
        three_r = entry + 3 * risk_per_share
    else:
        reward_per_share = entry - target
        one_r = entry - risk_per_share
        two_r = entry - 2 * risk_per_share
        three_r = entry - 3 * risk_per_share

    rr = reward_per_share / risk_per_share if risk_per_share else np.nan
    position_value = shares * entry
    return {
        "Dollar Risk": round(dollar_risk, 2),
        "Risk / Share": round(risk_per_share, 4),
        "Suggested Shares": int(max(shares, 0)),
        "Position Value": round(position_value, 2),
        "Potential Reward / Share": round(reward_per_share, 4),
        "R:R": round(rr, 2),
        "1R Level": round(one_r, 4),
        "2R Level": round(two_r, 4),
        "3R Level": round(three_r, 4),
    }


def score_trade(checks: Dict[str, bool]) -> Tuple[int, str]:
    weights = {
        "HTF bias aligned": 20,
        "Structure confirms direction": 20,
        "Valid POI / SND zone": 15,
        "MB / DMB confirmation": 15,
        "Wyckoff confirmation": 10,
        "Clean 1:3+ risk/reward": 10,
        "No major news / low-liquidity issue": 5,
        "Partial + breakeven plan": 5,
    }
    score = sum(weights[k] for k, v in checks.items() if v)
    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C / only if very clean"
    else:
        grade = "NO TRADE"
    return score, grade


def _journal_row(
    ticker: str,
    direction: str,
    timeframe: str,
    period: str,
    bias: str,
    bos: str,
    score: int,
    grade: str,
    risk_data: Dict[str, float | str],
    notes: str,
) -> pd.DataFrame:
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker.upper(),
        "Direction": direction,
        "Timeframe": timeframe,
        "Lookback": period,
        "Bias": bias,
        "BOS": bos,
        "Setup Score": score,
        "Setup Grade": grade,
        "Notes": notes,
    }
    for k, v in risk_data.items():
        row[k] = v
    return pd.DataFrame([row])


def render_smart_money_tab(default_ticker: str = "AAPL") -> None:
    st.subheader("Smart Money / M-Block Stock Analyzer")
    st.caption(
        "Rule-based planner built around Structure > SND > Wyckoff, M-Blocks, POIs, "
        "risk management, partials, and backtesting discipline."
    )

    with st.sidebar.expander("Smart Money Analyzer Settings", expanded=False):
        st.write("Use this tab to grade trade ideas, calculate risk, and journal your setups.")

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        ticker = st.text_input("Ticker", value=default_ticker).strip().upper()
    with col_b:
        period_label = st.selectbox("Lookback", list(PERIOD_OPTIONS.keys()), index=5)
    with col_c:
        interval_label = st.selectbox("Candle timeframe", list(INTERVAL_OPTIONS.keys()), index=6)

    period = PERIOD_OPTIONS[period_label]
    interval = INTERVAL_OPTIONS[interval_label]

    if not ticker:
        st.warning("Enter a ticker to begin.")
        return

    try:
        raw = fetch_ohlcv(ticker, period, interval)
    except Exception as exc:
        st.error(f"Could not fetch chart data: {exc}")
        return

    if raw.empty:
        st.error("No data returned. Try a different ticker, lookback, or timeframe.")
        return

    df = add_indicators(raw)
    df = detect_swings(df, window=3)
    df = detect_market_blocks(df)

    bias = classify_trend(df)
    bos_text, bos_level = detect_bos(df)
    zones_df = recent_zones(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
    m2.metric("Bias", bias)
    m3.metric("Structure", bos_text)
    m4.metric("Detected Zones", len(zones_df))

    fig = make_chart(df, ticker, zones_df)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Recent possible MB / SND zones", expanded=True):
        if zones_df.empty:
            st.info("No recent zones detected by the heuristic. Manually mark your POI from the chart.")
        else:
            st.dataframe(zones_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Trade Grade Checklist")

    st.write("The app should not force trades. It should keep you out unless the setup is clean.")

    c1, c2 = st.columns(2)
    with c1:
        htf = st.checkbox("HTF bias aligned", value=("Bullish" in bias or "Bearish" in bias))
        structure = st.checkbox("Structure confirms direction", value=("BOS" in bos_text))
        poi = st.checkbox("Valid POI / SND zone")
        mb = st.checkbox("MB / DMB confirmation")
    with c2:
        wyckoff = st.checkbox("Wyckoff confirmation")
        rr_clean = st.checkbox("Clean 1:3+ risk/reward")
        no_news = st.checkbox("No major news / low-liquidity issue", value=True)
        partial_plan = st.checkbox("Partial + breakeven plan", value=True)

    checks = {
        "HTF bias aligned": htf,
        "Structure confirms direction": structure,
        "Valid POI / SND zone": poi,
        "MB / DMB confirmation": mb,
        "Wyckoff confirmation": wyckoff,
        "Clean 1:3+ risk/reward": rr_clean,
        "No major news / low-liquidity issue": no_news,
        "Partial + breakeven plan": partial_plan,
    }

    score, grade = score_trade(checks)
    g1, g2 = st.columns(2)
    g1.metric("Setup Score", f"{score}/100")
    g2.metric("Setup Grade", grade)

    if grade == "NO TRADE":
        st.error("NO TRADE: setup does not have enough confirmation under your rules.")
    elif score < 75:
        st.warning("Low-to-medium quality setup. Consider waiting for cleaner confirmation.")
    else:
        st.success("Higher-quality setup based on the checklist. Still confirm manually before trading.")

    st.divider()
    st.subheader("Risk + Position Size Calculator")

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        direction = st.selectbox("Direction", ["Long", "Short"])
    with r2:
        account_size = st.number_input("Account Size", min_value=0.0, value=10000.0, step=500.0)
    with r3:
        risk_pct = st.number_input("Risk %", min_value=0.01, max_value=10.0, value=0.5, step=0.1)
    with r4:
        entry = st.number_input("Entry", min_value=0.0, value=float(df["Close"].iloc[-1]), step=0.01, format="%.4f")
    with r5:
        stop = st.number_input("Stop", min_value=0.0, value=float(df["Low"].tail(20).min()), step=0.01, format="%.4f")

    default_target = entry + 3 * abs(entry - stop) if direction == "Long" else max(entry - 3 * abs(entry - stop), 0.0)
    target = st.number_input("Target", min_value=0.0, value=float(default_target), step=0.01, format="%.4f")

    risk_data = calc_risk(TradeInputs(ticker, direction, account_size, risk_pct, entry, stop, target))
    if "error" in risk_data:
        st.error(str(risk_data["error"]))
    else:
        st.dataframe(pd.DataFrame([risk_data]), use_container_width=True, hide_index=True)

        if float(risk_data["R:R"]) >= 3:
            st.success("R:R passes the 1:3 target rule.")
        else:
            st.warning("R:R is below 1:3. Either improve entry, tighten invalidation, or skip.")

    st.divider()
    st.subheader("Trade Journal Export")

    notes = st.text_area(
        "Setup notes",
        placeholder=(
            "Example: HTF bullish, price retraced into demand MB, MB2 reconfirmed, "
            "possible reaccumulation, entry on confirmation, stop below MB low."
        ),
        height=100,
    )

    journal = _journal_row(
        ticker=ticker,
        direction=direction,
        timeframe=interval_label,
        period=period_label,
        bias=bias,
        bos=bos_text,
        score=score,
        grade=grade,
        risk_data=risk_data if "error" not in risk_data else {},
        notes=notes,
    )

    st.download_button(
        "Download this trade plan as CSV",
        data=journal.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_smart_money_trade_plan.csv",
        mime="text/csv",
    )

    with st.expander("What this analyzer is doing"):
        st.markdown(
            """
            This is a **semi-automatic trade planner**, not a black-box signal bot.

            It automatically helps with:
            - EMA alignment and rough trend context
            - Swing highs/lows
            - Body-close BOS/displacement checks
            - Potential demand/supply MB zones
            - Risk, R:R, 1R/2R/3R levels, and position size
            - Trade grading and journal export

            You still manually confirm:
            - Whether the MB is truly valid
            - Whether the POI is worth trading
            - Whether the setup matches your ACCMB → MMB execution model
            - Whether Wyckoff is actually present inside the zone
            """
        )
