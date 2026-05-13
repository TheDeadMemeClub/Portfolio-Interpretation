import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio_parser import parse_broker_upload, tradingview_csv
from market_data import fetch_fundamentals, fetch_returns, fetch_technical_snapshot, max_drawdown_from_returns

st.set_page_config(page_title="Portfolio EPIC", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    .epic-card {border: 1px solid rgba(255,255,255,.12); border-radius: 16px; padding: 16px; background: rgba(255,255,255,.03);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Portfolio EPIC")
st.caption("Upload a broker positions export and turn it into a private advisor-grade dashboard: allocation, heat maps, trend ratings, valuation, income, risk, tax lots, and exports.")


@st.cache_data(ttl=900, show_spinner=False)
def cached_technical_snapshot(symbols_tuple, period, interval):
    return fetch_technical_snapshot(list(symbols_tuple), period=period, interval=interval)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fundamentals(symbols_tuple):
    return fetch_fundamentals(list(symbols_tuple))


@st.cache_data(ttl=900, show_spinner=False)
def cached_returns(symbols_tuple, period):
    return fetch_returns(list(symbols_tuple), period=period)


def trend_badge_color(value):
    if value == "Bullish":
        return "background-color: rgba(0, 160, 70, .35); color: white; font-weight: 700"
    if value == "Bearish":
        return "background-color: rgba(200, 0, 0, .35); color: white; font-weight: 700"
    if value == "Neutral":
        return "background-color: rgba(180, 180, 180, .18); color: white"
    return ""

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Broker positions file", type=["xls", "xlsx", "csv"])
    st.caption("Optimized for Wells Fargo Advisors position exports. Your broker file is only used at runtime; it is not bundled in the app.")

    st.header("Dashboard controls")
    color_by = st.radio("Heat map color by", ["Today's Change %", "Unrealized P&L %", "Portfolio Weight"], horizontal=False)
    min_tile = st.slider("Hide tiny tiles below market value", 0, 10000, 0, 250)

    st.header("Market data")
    use_market_data = st.checkbox("Fetch live TA + valuation data", value=True)
    interval = st.selectbox("Primary technical candle size", ["1h", "4h", "1d", "1wk"], index=2)
    period = st.selectbox("Primary technical lookback", ["6mo", "1y", "2y", "5y"], index=1)
    build_mtf = st.checkbox("Build multi-timeframe trend matrix", value=True)
    benchmark = st.text_input("Benchmark", value="SPY")
    risk_free = st.number_input("Risk-free rate %", min_value=0.0, max_value=25.0, value=4.5, step=0.25) / 100

if uploaded is None:
    st.info("Upload your latest broker positions Excel file to generate the dashboard.")
    st.markdown(
        """
        **Workflow:** download positions from your broker → upload here → review allocation, heat map, trend ratings, valuations, risk flags, income, tax lots, and exports.

        **Privacy setup:** keep this repo public-safe by never committing `.xls`, `.xlsx`, or `.csv` broker files. The app starts empty and analyzes only the file you upload in the sidebar.
        """
    )
    st.stop()

try:
    result = parse_broker_upload(uploaded)
except Exception as e:
    st.error(f"Could not parse this file: {e}")
    st.stop()

summary = result.summary.copy()
lots = result.lots.copy()
allocation = result.allocation.copy()

# Core metrics
broker_total = result.broker_total if result.broker_total else allocation["Market Value"].sum()
ugl_total = summary["Unrealized P&L $"].fillna(0).sum()
today_total = summary["Today's Change $"].fillna(0).sum()
equity_count = summary.loc[summary["Asset Type"].eq("Stocks"), "Symbol"].nunique()
income_total = summary["Est. Annual Income"].fillna(0).sum()
securities_symbols = summary[summary["Asset Type"].isin(["Stocks", "ETFs", "Mutual Funds"])] ["Symbol"].dropna().astype(str).str.upper().unique().tolist()
market_symbols = summary[summary["Asset Type"].isin(["Stocks", "ETFs"])] ["Symbol"].dropna().astype(str).str.upper().unique().tolist()

# Optional live enrichments
tech = pd.DataFrame()
mtf_tech = pd.DataFrame()
fund = pd.DataFrame()
returns = pd.DataFrame()
if use_market_data:
    with st.spinner("Fetching market data, calculating EMA/200 MA trend ratings across candle timeframes, valuations, and risk metrics..."):
        symbol_tuple = tuple(securities_symbols)
        market_tuple = tuple(market_symbols)
        tech = cached_technical_snapshot(symbol_tuple, period, interval)
        if build_mtf:
            # Use enough history for each candle size so the 200 MA is meaningful.
            mtf_plan = [("1h", "6mo"), ("4h", "1y"), ("1d", "2y"), ("1wk", "5y")]
            frames = []
            for tf, tf_period in mtf_plan:
                frame = cached_technical_snapshot(symbol_tuple, tf_period, tf)
                if not frame.empty:
                    frames.append(frame)
            mtf_tech = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        fund = cached_fundamentals(market_tuple)
        returns = cached_returns(tuple(market_symbols + ([benchmark] if benchmark else [])), "1y")

if not tech.empty:
    summary = summary.merge(tech, on="Symbol", how="left")
if not fund.empty:
    summary = summary.merge(fund, on="Symbol", how="left")

# Derived advisor flags
summary["Portfolio Weight"] = summary["Portfolio Weight"].fillna(0)
summary["Risk Flag"] = ""
summary.loc[summary["Portfolio Weight"] > 0.10, "Risk Flag"] += "Oversized position; "
if "Trend Rating" in summary.columns:
    summary.loc[summary["Trend Rating"].eq("Bearish") & (summary["Portfolio Weight"] > 0.01), "Risk Flag"] += "Bearish trend; "
if "Drawdown from 52W High %" in summary.columns:
    summary.loc[summary["Drawdown from 52W High %"] < -0.25, "Risk Flag"] += "Deep drawdown; "
summary["Risk Flag"] = summary["Risk Flag"].str.rstrip("; ")

# Top row
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total portfolio value", f"${broker_total:,.0f}")
c2.metric("Unrealized P&L", f"${ugl_total:,.0f}", f"{ugl_total / broker_total:.2%}" if broker_total else None)
c3.metric("Today's change", f"${today_total:,.0f}", f"{today_total / broker_total:.2%}" if broker_total else None)
c4.metric("Equity positions", f"{equity_count} stocks")
c5.metric("Est. annual income", f"${income_total:,.0f}", f"{income_total / broker_total:.2%}" if broker_total else None)
c6.metric("Priced date", result.priced_date or "Not found")

# Executive flags
flag_items = []
cash_weight = float(allocation.loc[allocation["Asset Class"].eq("Cash"), "Market Value"].sum() / broker_total) if broker_total else 0
stock_weight = float(allocation.loc[allocation["Asset Class"].eq("Stocks"), "Market Value"].sum() / broker_total) if broker_total else 0
top1_weight = float(summary["Portfolio Weight"].max()) if not summary.empty else 0
top5_weight = float(summary.head(5)["Portfolio Weight"].sum()) if not summary.empty else 0
if cash_weight > 0.20:
    flag_items.append(f"Cash is high at {cash_weight:.1%}; decide whether it is dry powder or drag.")
if top1_weight > 0.12:
    flag_items.append(f"Largest holding is {top1_weight:.1%}; monitor single-name concentration.")
if top5_weight > 0.40:
    flag_items.append(f"Top 5 holdings are {top5_weight:.1%}; portfolio is concentrated.")
if "Trend Rating" in summary.columns:
    bearish_weight = summary.loc[summary["Trend Rating"].eq("Bearish"), "Portfolio Weight"].sum()
    if bearish_weight > 0.15:
        flag_items.append(f"Bearish technical exposure is {bearish_weight:.1%} based on EMA10/EMA20/EMA50 plus 200 MA structure.")
if not flag_items:
    flag_items.append("No major dashboard-level concentration or trend flags triggered by the current rules.")

with st.expander("Advisor cockpit: what needs attention first", expanded=True):
    for item in flag_items:
        st.write("• " + item)

st.divider()

# Main visuals
left, right = st.columns([1.05, 2.15])
with left:
    st.subheader("Asset allocation")
    fig_alloc = px.pie(allocation, names="Asset Class", values="Market Value", hole=0.52)
    fig_alloc.update_traces(textposition="inside", textinfo="percent+label")
    fig_alloc.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=385)
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.subheader("Allocation bar")
    fig_bar = px.bar(allocation.sort_values("Market Value", ascending=True), x="Market Value", y="Asset Class", orientation="h", text="Weight")
    fig_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_bar.update_layout(height=260, margin=dict(l=0, r=20, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Portfolio heat map")
    hm = summary[summary["Market Value"].fillna(0) >= min_tile].copy()
    if hm.empty:
        st.warning("No holdings passed the market value filter.")
    else:
        hover_cols = {
            "Description": True,
            "Market Value": ":$,.2f",
            "Portfolio Weight": ":.2%",
            "Unrealized P&L $": ":$,.2f",
            "Unrealized P&L %": ":.2%",
            "Today's Change $": ":$,.2f",
            "Today's Change %": ":.2%",
        }
        if "Trend Rating" in hm.columns:
            hover_cols.update({"Trend Rating": True, "Trend Setup": True, "Distance from 50 EMA %": ":.2%"})
        fig = px.treemap(
            hm,
            path=["Asset Type", "Symbol"],
            values="Market Value",
            color=color_by,
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0 if color_by != "Portfolio Weight" else hm[color_by].median(),
            hover_data=hover_cols,
        )
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:$,.0f}")
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=20), height=650)
        st.plotly_chart(fig, use_container_width=True)

# Tabs
labels = ["Holdings", "Trend Ratings", "Valuation", "Risk", "Winners / Losers", "Income", "Tax Lots", "Rebalance", "Exports"]
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(labels)

with tab1:
    st.subheader("Cleaned holdings summary")
    base_cols = ["Symbol", "Description", "Asset Type", "Shares", "Last Price", "Market Value", "Portfolio Weight", "Total Cost", "Avg Cost", "Unrealized P&L $", "Unrealized P&L %", "Today's Change $", "Today's Change %", "Est. Annual Income", "Yield on MV", "Lot Count", "Risk Flag"]
    extra_cols = [c for c in ["Trend Rating", "Sector", "Industry", "Beta", "Trailing P/E", "Forward P/E"] if c in summary.columns]
    st.dataframe(summary[[c for c in base_cols + extra_cols if c in summary.columns]], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("EMA + 200 MA trend rating engine")
    st.write("Bullish = **EMA10 > EMA20 > EMA50 and Close > 200 MA**. Bearish = **EMA10 < EMA20 < EMA50 and Close < 200 MA**. Everything else is Neutral.")
    st.caption(f"Primary technical table is currently using **{interval} candles**. Change that in the sidebar. The multi-timeframe matrix shows hourly, 4-hour, daily, and weekly at the same time.")
    if tech.empty:
        st.warning("Turn on market data in the sidebar to calculate live technical ratings.")
    else:
        tcols = ["Symbol", "Trend Rating", "Trend Setup", "Last Close", "EMA 10", "EMA 20", "EMA 50", "MA 200", "Distance from 50 EMA %", "Distance from 200 MA %", "Drawdown from 52W High %"]
        tech_view = summary[[c for c in ["Symbol", "Description", "Asset Type", "Market Value", "Portfolio Weight"] + tcols[1:] if c in summary.columns]].copy()
        rating_counts = tech_view.groupby("Trend Rating", dropna=False)["Portfolio Weight"].sum().reset_index()
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric(f"Bullish exposure ({interval})", f"{rating_counts.loc[rating_counts['Trend Rating'].eq('Bullish'), 'Portfolio Weight'].sum():.1%}")
            st.metric(f"Bearish exposure ({interval})", f"{rating_counts.loc[rating_counts['Trend Rating'].eq('Bearish'), 'Portfolio Weight'].sum():.1%}")
            st.metric(f"Neutral/no-data exposure ({interval})", f"{rating_counts.loc[~rating_counts['Trend Rating'].isin(['Bullish','Bearish']), 'Portfolio Weight'].sum():.1%}")
        with colB:
            fig_rating = px.bar(rating_counts, x="Trend Rating", y="Portfolio Weight", text="Portfolio Weight", title=f"Portfolio exposure by trend rating — {interval} candles")
            fig_rating.update_traces(texttemplate="%{text:.1%}")
            fig_rating.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_rating, use_container_width=True)

        if not mtf_tech.empty:
            st.markdown("### Multi-timeframe trend matrix")
            mtf = mtf_tech.merge(summary[["Symbol", "Description", "Asset Type", "Market Value", "Portfolio Weight"]], on="Symbol", how="left")
            tf_order = ["1h", "4h", "1d", "1wk"]
            matrix = mtf.pivot_table(index=["Symbol", "Description", "Asset Type", "Portfolio Weight"], columns="Timeframe", values="Trend Rating", aggfunc="first").reset_index()
            for tf in tf_order:
                if tf not in matrix.columns:
                    matrix[tf] = "No data"
            matrix["Alignment Score"] = matrix[tf_order].apply(lambda r: sum(1 if x == "Bullish" else -1 if x == "Bearish" else 0 for x in r), axis=1)
            matrix["Read"] = np.select(
                [matrix["Alignment Score"].ge(3), matrix["Alignment Score"].le(-3), matrix[["1d", "1wk"]].eq("Bullish").all(axis=1), matrix[["1d", "1wk"]].eq("Bearish").all(axis=1)],
                ["Bullish across most timeframes", "Bearish across most timeframes", "Higher-timeframe bullish", "Higher-timeframe bearish"],
                default="Mixed / timeframe conflict"
            )
            st.dataframe(
                matrix[["Symbol", "Description", "Asset Type", "Portfolio Weight"] + tf_order + ["Alignment Score", "Read"]]
                .sort_values(["Alignment Score", "Portfolio Weight"], ascending=[False, False])
                .style.applymap(trend_badge_color, subset=tf_order),
                use_container_width=True,
                hide_index=True,
            )

            mtf_exposure = mtf.groupby(["Timeframe", "Trend Rating"], dropna=False)["Portfolio Weight"].sum().reset_index()
            fig_mtf = px.bar(mtf_exposure, x="Timeframe", y="Portfolio Weight", color="Trend Rating", text="Portfolio Weight", category_orders={"Timeframe": tf_order}, title="Bullish / bearish / neutral exposure by candle timeframe")
            fig_mtf.update_traces(texttemplate="%{text:.1%}")
            fig_mtf.update_layout(height=360, margin=dict(l=0, r=0, t=45, b=20))
            st.plotly_chart(fig_mtf, use_container_width=True)

            st.markdown("### Detailed multi-timeframe EMA table")
            detailed_cols = ["Symbol", "Timeframe", "Trend Rating", "Trend Setup", "Last Close", "EMA 10", "EMA 20", "EMA 50", "MA 200", "Distance from 200 MA %", "Drawdown from 52W High %"]
            st.dataframe(mtf[[c for c in detailed_cols if c in mtf.columns]].sort_values(["Symbol", "Timeframe"]), use_container_width=True, hide_index=True)

        st.markdown("### Primary timeframe details")
        st.dataframe(tech_view.sort_values(["Trend Rating", "Portfolio Weight"], ascending=[True, False]), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Valuation + fundamentals")
    if fund.empty:
        st.warning("Turn on market data in the sidebar to pull valuation/fundamental data.")
    else:
        vcols = ["Symbol", "Description", "Sector", "Industry", "Market Value", "Portfolio Weight", "Market Cap", "Trailing P/E", "Forward P/E", "Price/Sales", "Price/Book", "EV/EBITDA", "Dividend Yield", "Beta", "Analyst Target Mean", "Recommendation", "Market Data Status"]
        val = summary[[c for c in vcols if c in summary.columns]].copy()
        st.dataframe(val.sort_values("Portfolio Weight", ascending=False), use_container_width=True, hide_index=True)
        sector = val.dropna(subset=["Sector"]).groupby("Sector", as_index=False)["Market Value"].sum()
        if not sector.empty:
            fig_sector = px.treemap(sector, path=["Sector"], values="Market Value", title="Equity/ETF sector exposure from market data")
            fig_sector.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=10))
            st.plotly_chart(fig_sector, use_container_width=True)

with tab4:
    st.subheader("Portfolio risk dashboard")
    hhi = float((summary["Portfolio Weight"].fillna(0) ** 2).sum())
    eff_n = 1 / hhi if hhi else np.nan
    beta_weighted = np.nan
    if "Beta" in summary.columns:
        beta_weighted = (summary["Beta"].fillna(0) * summary["Portfolio Weight"].fillna(0)).sum()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Largest position", f"{top1_weight:.1%}")
    r2.metric("Top 5 weight", f"{top5_weight:.1%}")
    r3.metric("Effective # holdings", f"{eff_n:.1f}" if pd.notna(eff_n) else "N/A")
    r4.metric("Weighted beta", f"{beta_weighted:.2f}" if pd.notna(beta_weighted) else "N/A")

    if returns.empty:
        st.warning("Turn on market data to estimate volatility, drawdown, and correlations.")
    else:
        # Build approximate weighted daily portfolio return from market symbols only.
        weights = summary.set_index("Symbol")["Portfolio Weight"].reindex(returns.columns).fillna(0)
        if weights.sum() > 0:
            port_ret = returns.mul(weights, axis=1).sum(axis=1)
            ann_vol = port_ret.std() * np.sqrt(252)
            ann_ret = port_ret.mean() * 252
            sharpe = (ann_ret - risk_free) / ann_vol if ann_vol and pd.notna(ann_vol) else np.nan
            dd = max_drawdown_from_returns(port_ret)
            rr1, rr2, rr3, rr4 = st.columns(4)
            rr1.metric("1Y est. annual return", f"{ann_ret:.1%}")
            rr2.metric("1Y est. volatility", f"{ann_vol:.1%}")
            rr3.metric("Est. Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
            rr4.metric("Max drawdown", f"{dd:.1%}" if pd.notna(dd) else "N/A")

        corr = returns[[c for c in returns.columns if c != benchmark]].corr()
        if corr.shape[0] >= 2:
            fig_corr = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="1Y daily return correlation matrix")
            fig_corr.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Biggest unrealized winners")
        st.dataframe(summary.sort_values("Unrealized P&L $", ascending=False).head(15)[["Symbol", "Market Value", "Unrealized P&L $", "Unrealized P&L %", "Portfolio Weight"]], use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Biggest unrealized losers")
        st.dataframe(summary.sort_values("Unrealized P&L $", ascending=True).head(15)[["Symbol", "Market Value", "Unrealized P&L $", "Unrealized P&L %", "Portfolio Weight"]], use_container_width=True, hide_index=True)

    st.subheader("Today's movers")
    st.dataframe(summary.sort_values("Today's Change $", ascending=False)[["Symbol", "Market Value", "Today's Change $", "Today's Change %", "Portfolio Weight"]].head(20), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("Income dashboard")
    income_df = summary[summary["Est. Annual Income"].fillna(0) > 0].sort_values("Est. Annual Income", ascending=False)
    st.metric("Estimated annual income from securities", f"${income_df['Est. Annual Income'].sum():,.0f}")
    if not income_df.empty:
        fig_income = px.bar(income_df.head(25), x="Symbol", y="Est. Annual Income", hover_data=["Description", "Market Value", "Yield on MV"])
        fig_income.update_layout(height=430, margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_income, use_container_width=True)
        st.dataframe(income_df[["Symbol", "Description", "Asset Type", "Market Value", "Est. Annual Income", "Yield on MV", "Portfolio Weight"]], use_container_width=True, hide_index=True)

with tab7:
    st.subheader("Tax-lot / source-row cleanup")
    st.write("The parser avoids double-counting broker summary rows and keeps usable lot rows for trade-date and basis analysis.")
    if lots.empty:
        st.warning("No lot rows found.")
    else:
        st.dataframe(lots, use_container_width=True, hide_index=True)

with tab8:
    st.subheader("Simple rebalance sandbox")
    st.write("Set rough target weights by asset class. This estimates buy/sell dollar amounts against the current portfolio value.")
    target_defaults = {"Stocks": 35.0, "ETFs": 15.0, "Mutual Funds": 25.0, "Cash": 20.0, "Fixed Income": 5.0}
    targets = {}
    cols = st.columns(len(target_defaults))
    for i, asset in enumerate(target_defaults):
        with cols[i]:
            targets[asset] = st.number_input(f"{asset} target %", 0.0, 100.0, target_defaults[asset], 1.0) / 100
    total_target = sum(targets.values())
    if abs(total_target - 1) > 0.01:
        st.warning(f"Targets currently add to {total_target:.1%}. Make them total about 100%.")
    reb = allocation.copy()
    reb["Current Weight"] = reb["Market Value"] / broker_total if broker_total else np.nan
    reb["Target Weight"] = reb["Asset Class"].map(targets).fillna(0)
    reb["Target Value"] = reb["Target Weight"] * broker_total
    reb["Buy / Sell $"] = reb["Target Value"] - reb["Market Value"]
    st.dataframe(reb, use_container_width=True, hide_index=True)

with tab9:
    st.subheader("Download cleaned/enriched files")
    summary_csv = summary.to_csv(index=False).encode("utf-8")
    lots_csv = lots.to_csv(index=False).encode("utf-8")
    alloc_csv = allocation.to_csv(index=False).encode("utf-8")
    tv_consolidated = tradingview_csv(result.summary, "consolidated").to_csv(index=False).encode("utf-8")
    tv_lot = tradingview_csv(lots, "lot").to_csv(index=False).encode("utf-8") if not lots.empty else b""

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download enriched holdings CSV", summary_csv, "portfolio_holdings_enriched.csv", "text/csv")
    d2.download_button("Download tax lots CSV", lots_csv, "portfolio_lot_level_cleaned.csv", "text/csv")
    d3.download_button("Download allocation CSV", alloc_csv, "portfolio_allocation.csv", "text/csv")

    d4, d5 = st.columns(2)
    d4.download_button("Download TradingView-style consolidated CSV", tv_consolidated, "tradingview_portfolio_import_consolidated.csv", "text/csv")
    if tv_lot:
        d5.download_button("Download TradingView-style lot CSV", tv_lot, "tradingview_portfolio_import_lot_level.csv", "text/csv")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("portfolio_holdings_enriched.csv", summary_csv)
        z.writestr("portfolio_lot_level_cleaned.csv", lots_csv)
        z.writestr("portfolio_allocation.csv", alloc_csv)
        z.writestr("tradingview_portfolio_import_consolidated.csv", tv_consolidated)
        if tv_lot:
            z.writestr("tradingview_portfolio_import_lot_level.csv", tv_lot)
    st.download_button("Download all outputs as ZIP", zip_buffer.getvalue(), "portfolio_epic_outputs.zip", "application/zip")
