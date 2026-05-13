from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NUMERIC_FIELDS = [
    "Shares", "Market Value", "Total Cost1", "Original Cost", "Total Client Investment",
    "Unrealized Gain/Loss ($)1", "Client Inv Gain/(Loss) $", "Est. Annual Income",
    "Today's Change ($)1", "Change from Prev ($)", "Last Price ($)", "Trade Price",
    "Cost Basis"
]


@dataclass
class PortfolioParseResult:
    summary: pd.DataFrame
    lots: pd.DataFrame
    allocation: pd.DataFrame
    priced_date: str | None
    broker_total: float | None
    cash_total: float
    fixed_income_total: float


def clean_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in {"", "--", "N/A", "nan", "None"}:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("+", "")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


def clean_date(x, fallback: str | None = None):
    if pd.isna(x) or str(x).strip() in {"", "Detail", "N/A", "Intra-Day", "Detailnc"}:
        return fallback or ""
    s = str(x).replace("nc", "").strip()
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return s


def read_broker_file(uploaded_file) -> pd.DataFrame:
    """Read xls/xlsx/csv into a raw no-header dataframe."""
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, header=None, dtype=object)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, header=None, dtype=object, engine="openpyxl")
    if name.endswith(".xls"):
        # Wells Fargo exports usually work with xlrd. requirements.txt installs it.
        return pd.read_excel(uploaded_file, header=None, dtype=object, engine="xlrd")
    # fallback: try Excel first, then CSV
    try:
        return pd.read_excel(uploaded_file, header=None, dtype=object)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, header=None, dtype=object)


def _row_text(df: pd.DataFrame, row_idx: int) -> str:
    if row_idx < 0 or row_idx >= len(df):
        return ""
    return " ".join(str(x) for x in df.iloc[row_idx].dropna().tolist())


def extract_priced_date(df: pd.DataFrame) -> str | None:
    for i in range(min(15, len(df))):
        txt = _row_text(df, i)
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", txt)
        if m:
            return clean_date(m.group(1))
    return None


def extract_summary_totals(df: pd.DataFrame) -> Tuple[float, float, float | None]:
    cash_total = 0.0
    fixed_income_total = 0.0
    broker_total = None
    for i in range(len(df)):
        cells = df.iloc[i].tolist()
        text_cells = [str(x).strip() for x in cells if pd.notna(x)]
        joined = " ".join(text_cells)
        values = [clean_number(x) for x in cells]
        numeric_values = [v for v in values if pd.notna(v)]
        if "Total Cash" in joined and numeric_values:
            cash_total = float(numeric_values[0])
        if "Total Fixed Income" in joined and numeric_values:
            # Usually the market value is the largest usable number on the row.
            fixed_income_total = float(max(numeric_values, key=abs))
        if "Total Portfolio" in joined and numeric_values:
            broker_total = float(numeric_values[0])
    return cash_total, fixed_income_total, broker_total


def find_position_sections(df: pd.DataFrame) -> List[Tuple[str, int, int, int]]:
    """
    Returns tuples: (asset_type, header_row, start_row, end_row).
    Dynamically locates WFA-like section headers instead of hardcoding rows.
    """
    section_names = ["Stocks", "ETFs", "Mutual Funds"]
    section_markers: List[Tuple[str, int]] = []
    for i in range(len(df)):
        row_vals = [str(x).strip() for x in df.iloc[i].dropna().tolist()]
        for section in section_names:
            if any(v == section for v in row_vals):
                section_markers.append((section, i))

    sections = []
    for idx, (section, marker_row) in enumerate(section_markers):
        next_marker = section_markers[idx + 1][1] if idx + 1 < len(section_markers) else len(df)
        header_row = None
        for r in range(marker_row, min(marker_row + 6, len(df))):
            vals = [str(x).strip() for x in df.iloc[r].tolist()]
            if "Symbol" in vals and "Description" in vals:
                header_row = r
                break
        if header_row is None:
            continue
        start = header_row + 1
        end = next_marker - 1
        sections.append((section, header_row, start, end))
    return sections


def parse_positions(df: pd.DataFrame) -> Tuple[pd.DataFrame, str | None, float, float, float | None]:
    priced_date = extract_priced_date(df)
    cash_total, fixed_income_total, broker_total = extract_summary_totals(df)
    sections = find_position_sections(df)
    rows = []

    for asset_type, header_row, start, end in sections:
        headers = [str(x).strip() if pd.notna(x) else "" for x in df.iloc[header_row].tolist()]
        for r in range(start, end + 1):
            raw_vals = df.iloc[r].tolist()
            # Skip blank/total/category rows.
            joined = " ".join(str(x).strip() for x in raw_vals if pd.notna(x))
            if not joined or joined.startswith("Total"):
                continue
            if any(joined.startswith(prefix) for prefix in ["Common Stock", "Money Market", "Open End", "Closed End"]):
                continue
            rec = {h: raw_vals[c] for c, h in enumerate(headers) if h}
            symbol = str(rec.get("Symbol", "")).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            rec["Symbol"] = symbol.upper()
            rec["Asset Type"] = asset_type
            rec["Source Row"] = r + 1
            rows.append(rec)

    positions = pd.DataFrame(rows)
    for col in NUMERIC_FIELDS:
        if col in positions.columns:
            positions[col] = positions[col].apply(clean_number)
    return positions, priced_date, cash_total, fixed_income_total, broker_total


def build_analysis(positions: pd.DataFrame, priced_date: str | None, cash_total: float, fixed_income_total: float, broker_total: float | None) -> PortfolioParseResult:
    if positions.empty:
        raise ValueError("No position rows found. This app currently expects a Wells Fargo-style positions export with Stocks, ETFs, and/or Mutual Funds sections.")

    summary_rows = []
    for sym, group in positions.groupby("Symbol", dropna=True):
        details = group[group.get("Tax Term", pd.Series(index=group.index, dtype=object)).astype(str).eq("Detail")]
        use = details if not details.empty else group
        base = use.iloc[0]

        def sum_first(*cols):
            for col in cols:
                if col in use.columns and use[col].notna().any():
                    return float(use[col].sum(skipna=True))
            return np.nan

        shares = sum_first("Shares")
        mv = sum_first("Market Value")
        cost = sum_first("Total Cost1", "Original Cost", "Total Client Investment")
        ugl = sum_first("Unrealized Gain/Loss ($)1", "Client Inv Gain/(Loss) $")
        today = sum_first("Today's Change ($)1", "Change from Prev ($)")
        income = sum_first("Est. Annual Income")
        last_price = base.get("Last Price ($)", np.nan)

        summary_rows.append({
            "Symbol": sym,
            "Description": base.get("Description", ""),
            "Asset Type": base.get("Asset Type", ""),
            "Shares": shares,
            "Last Price": last_price,
            "Market Value": mv,
            "Total Cost": cost,
            "Avg Cost": cost / shares if pd.notna(cost) and pd.notna(shares) and shares else np.nan,
            "Unrealized P&L $": ugl,
            "Unrealized P&L %": ugl / cost if pd.notna(ugl) and pd.notna(cost) and cost else np.nan,
            "Today's Change $": today,
            "Today's Change %": today / mv if pd.notna(today) and pd.notna(mv) and mv else np.nan,
            "Est. Annual Income": income,
            "Yield on MV": income / mv if pd.notna(income) and pd.notna(mv) and mv else np.nan,
            "Lot Count": int(len(group)),
        })

    summary = pd.DataFrame(summary_rows)
    securities_total = summary["Market Value"].fillna(0).sum()
    total_for_weights = broker_total if broker_total and broker_total > 0 else securities_total + cash_total + fixed_income_total
    summary["Portfolio Weight"] = summary["Market Value"] / total_for_weights if total_for_weights else np.nan
    summary = summary.sort_values("Market Value", ascending=False, na_position="last")

    lot_rows = []
    for sym, group in positions.groupby("Symbol", dropna=True):
        has_detail = "Tax Term" in group.columns and group["Tax Term"].astype(str).eq("Detail").any()
        for _, p in group.iterrows():
            if has_detail and str(p.get("Tax Term", "")) == "Detail":
                continue
            q = p.get("Shares", np.nan)
            if pd.isna(q) or q == 0:
                continue
            fill = p.get("Trade Price", np.nan)
            if pd.isna(fill):
                fill = p.get("Cost Basis", np.nan)
            total_cost = p.get("Total Cost1", np.nan)
            if pd.isna(fill) and pd.notna(total_cost):
                fill = total_cost / q if q else np.nan
            lot_rows.append({
                "Symbol": sym,
                "Description": p.get("Description", ""),
                "Asset Type": p.get("Asset Type", ""),
                "Side": "Buy" if q > 0 else "Sell",
                "Qty": abs(q),
                "Signed Qty": q,
                "Fill Price": fill,
                "Commission": 0,
                "Closing Time": clean_date(p.get("Trade Date1", None), priced_date),
                "Market Value": p.get("Market Value", np.nan),
                "Total Cost": total_cost if pd.notna(total_cost) else p.get("Original Cost", np.nan),
                "Unrealized P&L $": p.get("Unrealized Gain/Loss ($)1", np.nan),
                "Tax Term": p.get("Tax Term", ""),
                "Source Row": p.get("Source Row", ""),
            })
    lots = pd.DataFrame(lot_rows)

    allocation_parts = [
        ("Stocks", summary.loc[summary["Asset Type"].eq("Stocks"), "Market Value"].fillna(0).sum()),
        ("ETFs", summary.loc[summary["Asset Type"].eq("ETFs"), "Market Value"].fillna(0).sum()),
        ("Mutual Funds", summary.loc[summary["Asset Type"].eq("Mutual Funds"), "Market Value"].fillna(0).sum()),
        ("Cash", cash_total),
        ("Fixed Income", fixed_income_total),
    ]
    allocation = pd.DataFrame(allocation_parts, columns=["Asset Class", "Market Value"])
    allocation = allocation[allocation["Market Value"].fillna(0) > 0].copy()
    alloc_total = allocation["Market Value"].sum()
    allocation["Weight"] = allocation["Market Value"] / alloc_total if alloc_total else np.nan

    return PortfolioParseResult(summary, lots, allocation, priced_date, broker_total, cash_total, fixed_income_total)


def parse_broker_upload(uploaded_file) -> PortfolioParseResult:
    df = read_broker_file(uploaded_file)
    positions, priced_date, cash_total, fixed_income_total, broker_total = parse_positions(df)
    return build_analysis(positions, priced_date, cash_total, fixed_income_total, broker_total)


def tradingview_csv(summary_or_lots: pd.DataFrame, mode: str = "consolidated") -> pd.DataFrame:
    if mode == "lot" and {"Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"}.issubset(summary_or_lots.columns):
        return summary_or_lots[["Symbol", "Side", "Qty", "Fill Price", "Commission", "Closing Time"]].copy()
    df = summary_or_lots.copy()
    out = pd.DataFrame({
        "Symbol": df["Symbol"],
        "Side": np.where(df["Shares"].fillna(0) >= 0, "Buy", "Sell"),
        "Qty": df["Shares"].abs(),
        "Fill Price": df["Avg Cost"].fillna(df["Last Price"]),
        "Commission": 0,
        "Closing Time": pd.Timestamp.today().strftime("%Y-%m-%d"),
    })
    return out
