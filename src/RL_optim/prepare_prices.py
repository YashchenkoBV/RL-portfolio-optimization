#!/usr/bin/env python3
import csv
import pandas as pd
import yfinance as yf

def build_price_splits(in_csv="data/company_prices.csv"):
    # tickers are on 3rd line (index 2), from 2nd column (index 1)
    with open(in_csv, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        for _ in range(2): next(r)
        row = next(r)
    tickers = [c.strip() for c in row[1:] if c.strip()]
    # drop obvious non-tickers if present
    tickers = [t for t in tickers if t.lower() not in {"average","avg","std- dev-","std-dev","std dev","stdev","std"}]
    # Yahoo quirk: BRK.B -> BRK-B (keep exchange suffixes like VOD.L intact)
    tickers = [t[:-2] + "-" + t[-1] if t.count(".")==1 and t.split(".")[-1] in {"A","B"} else t for t in tickers]

    alias = {"GOOGLE": "GOOG", "FB": "META"}
    drop = {"ATVI", "JWN", "META", "MA", "V", "AVGO", "GM", "AZN", "ABBV"}
    out = []
    seen = set()
    for t in tickers:
        u = t.upper()
        t = alias.get(u, u)
        if t in drop:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    tickers = out

    for name, start, end in [("train", "2005-01-01", "2016-12-31"),
                             ("val", "2017-01-01", "2019-12-31"),
                             ("test", "2020-01-01", "2024-12-31")]:
        end_excl = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        close = yf.download(tickers, start=start, end=end_excl, auto_adjust=True, progress=False)["Close"]
        if isinstance(close, pd.Series): close = close.to_frame(tickers[0])
        out = close.reindex(columns=tickers).reset_index()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(f"price_{name}_data.csv", index=False)

if __name__ == "__main__":
    build_price_splits()
