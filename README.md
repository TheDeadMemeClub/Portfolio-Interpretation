# Portfolio EPIC

Private Streamlit dashboard for converting a broker positions export into an advisor-grade portfolio analysis tool.

## Core workflow

1. Download your positions export from your broker.
2. Upload the `.xls`, `.xlsx`, or `.csv` file inside the Streamlit sidebar.
3. Review allocation, heat map, holdings, tax lots, income, risk, valuation, technical ratings, and exports.

## Technical rating rule

Bullish = `EMA10 > EMA20 > EMA50` and price/close is above the `200 MA`.

Bearish = `EMA10 < EMA20 < EMA50` and price/close is below the `200 MA`.

Anything else = Neutral.

The app supports a primary candle-size selector and a multi-timeframe trend matrix for `1h`, `4h`, `1d`, and `1wk` candles so you can see when a holding is bullish on the weekly chart but bearish on the hourly chart.

## Privacy

Do not commit broker files, exports, or CSVs to GitHub. The included `.gitignore` blocks common private portfolio files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
