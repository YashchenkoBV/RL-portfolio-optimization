"""
Plot daily (annualized) risk‑free rates from 2000 to 2025 from two sources:
- Fama‑French 1‑Month T‑Bill (daily RF) — annualized from daily returns
- FRED 1‑Month Treasury Constant Maturity Rate (DGS1MO), with DGS3MO as fallback

Requirements: yfinance, pandas, matplotlib (standard library only otherwise)

Notes
-----
* Fama‑French daily RF is given as a DAILY return in percent. We convert it to an
  ANNUALIZED rate via compound scaling: (1 + rf_daily)^252 − 1.
* FRED DGS1MO/DGS3MO are already quoted as annualized yields (percent), so they
  are used as‑is.
* Date range is clipped to [2000‑01‑01, min(2025‑12‑31, today)].
"""

from __future__ import annotations

import io
import re
import sys
import math
import urllib.request
from datetime import datetime
from zipfile import ZipFile

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
START_DATE = pd.Timestamp("2000-01-01")
END_DATE_CAP = pd.Timestamp("2025-12-31")
TODAY = pd.Timestamp.today().normalize()
END_DATE = min(END_DATE_CAP, TODAY)

# ----------------------------
# Helpers
# ----------------------------

def _read_ken_french_daily_rf() -> pd.DataFrame:
    """Download and robustly parse the F-F Research Data Factors (Daily) CSV.

    Strategy:
      • find first data row by detecting a yyyymmdd at line start,
      • use a nearby header if present (else inject a standard header),
      • normalize separators,
      • return a DataFrame indexed by datetime with numeric columns.
    """
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    )

    with urllib.request.urlopen(url) as resp:
        zf = ZipFile(io.BytesIO(resp.read()))
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw_text = zf.read(csv_name).decode("latin1", errors="ignore")

    lines = raw_text.splitlines()

    # 1) first/last data lines (YYYYMMDD at line start)
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d{8}\s*(,|;|\s)", line):
            data_start = i
            break
    if data_start is None:
        raise RuntimeError("Could not locate the first data row in Ken French CSV")

    data_end = data_start
    while data_end < len(lines) and re.match(r"^\s*\d{8}\s*(,|;|\s)", lines[data_end]):
        data_end += 1

    # 2) look for a plausible header above the data block
    header_idx = None
    for j in range(data_start - 1, max(-1, data_start - 25), -1):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if ("RF" in candidate or "Rf" in candidate or "rf" in candidate) and (
            "," in candidate or ";" in candidate or re.search(r"\s{2,}", candidate)
        ):
            header_idx = j
            break

    def _normalize_row(s: str) -> str:
        s = s.strip().replace(";", ",")
        if "," not in s:  # whitespace-separated -> commas
            s = re.sub(r"\s+", ",", s)
        return s

    if header_idx is not None:
        header = _normalize_row(lines[header_idx])
        header = re.sub(r"^(DATE|date)\b", "Date", header)
    else:
        header = "Date,Mkt-RF,SMB,HML,RF"

    data_block = [_normalize_row(ln) for ln in lines[data_start:data_end]]
    csv_text = "\n".join([header] + data_block)

    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    for col in df.columns:
        if col != df.index.name:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_fama_french_rf_annualized(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return a daily series of annualized RF (%) from Fama‑French daily data.

    The input RF is a daily return in percent. We convert to decimal per‑day,
    compound to annual using 252 trading days, then return as percent.
    """
    factors = _read_ken_french_daily_rf()
    rf_daily_pct = factors.loc[start:end, "RF"].dropna()  # daily %
    rf_daily_dec = rf_daily_pct / 100.0
    # Annualize: (1 + r_d)^252 - 1 (still a daily series, expressed as an annualized rate)
    rf_annualized_dec = (1.0 + rf_daily_dec) ** 252 - 1.0
    rf_annualized_pct = rf_annualized_dec * 100.0
    rf_annualized_pct.name = "Fama-French RF (annualized from daily)"
    return rf_annualized_pct


def _try_yf_fred_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    """Try to fetch a FRED series by symbol via yfinance (if Yahoo hosts it).

    Returns a Series of percent values (Close) if successful, else None.
    """
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
        if hist is not None and not hist.empty:
            # yfinance often exposes macro/FRED series in the 'Close' column
            col = "Close" if "Close" in hist.columns else hist.columns[0]
            s = hist[col].rename(symbol).astype(float)
            s.index = pd.to_datetime(s.index)
            return s
    except Exception:
        return None
    return None


def _read_fred_csv(symbol: str) -> pd.Series:
    """Read a FRED series via the fredgraph CSV endpoint with robust error checks."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed to download FRED series {symbol}: {e}") from e

    # Guard: FRED may return an HTML error/404 page; verify header contains DATE
    first_line = text.splitlines()[0] if text else ""
    if "DATE" not in first_line.upper():
        raise RuntimeError(
            f"Unexpected response when downloading {symbol} from FRED; got header: {first_line[:80]!r}"
        )

    df = pd.read_csv(io.StringIO(text))
    if "DATE" not in df.columns or symbol not in df.columns:
        raise RuntimeError(f"FRED CSV for {symbol} missing expected columns")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.set_index("DATE").sort_index()
    s = pd.to_numeric(df[symbol].replace(".", pd.NA), errors="coerce")
    s.name = symbol
    return s


def get_fred_tbill_annualized(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Get a daily annualized risk-free yield from public sources without an API key.

    Priority: FRED DGS1MO → FRED DGS3MO → FRED DTB3 → Yahoo ^IRX (13-week T-bill).
    Values are already annualized percent.
    """
    series_list: list[pd.Series] = []

    for sym in ["DGS1MO", "DGS3MO", "DTB3"]:
        try:
            s = _read_fred_csv(sym).loc[start:end]
            if not s.empty:
                series_list.append(s.rename(sym))
        except Exception:
            pass  # try next fallback

    # Yahoo fallback (13-week T-Bill). yfinance returns this in percent already.
    try:
        irx = _try_yf_fred_series("^IRX", start, end)
        if irx is not None and not irx.empty:
            series_list.append(irx.rename("^IRX"))
    except Exception:
        pass

    if not series_list:
        raise RuntimeError("Unable to retrieve any T-bill series from FRED or Yahoo.")

    fred = series_list[0]
    for s in series_list[1:]:
        fred = fred.combine_first(s)

    fred = pd.to_numeric(fred, errors="coerce").dropna()
    fred.name = "FRED T-Bill Yield (% annual)"
    return fred


# ----------------------------
# Main plotting
# ----------------------------
if __name__ == "__main__":
    print(f"Date window: {START_DATE.date()} → {END_DATE.date()}")

    # --- Fama-French ---
    ff_rf_ann = get_fama_french_rf_annualized(START_DATE, END_DATE)

    plt.figure(figsize=(10, 4.5))
    ff_rf_ann.plot()
    plt.title("Daily Annualized Risk‑Free Rate — FRED RF (1M T‑Bill)")
    plt.ylabel("Annualized rate (%)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

    # --- FRED ---
    fred_rf = get_fred_tbill_annualized(START_DATE, END_DATE)

    plt.figure(figsize=(10, 4.5))
    fred_rf.plot()
    plt.title("Daily Annualized Risk‑Free Rate — FRED (DGS1MO, fallback DGS3MO)")
    plt.ylabel("Annualized rate (%)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

    # Optional: print a quick alignment summary
    print("\nObservations:")
    print(f"  Fama-French RF (annualized) obs: {ff_rf_ann.size:,}")
    print(f"  FRED T-Bill yield obs:           {fred_rf.size:,}")
