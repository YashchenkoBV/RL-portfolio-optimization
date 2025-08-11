import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr


def fetch_prices(symbol, start="2000-01-01", end="2015-12-31") -> pd.Series:
    """
    Fetch Yahoo 'Adj Close' for an ETF/stock and return price series.
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol} [{start}→{end}]. ")

    cl = df["Close"].squeeze("columns")  # 1-col DF -> Series
    price = cl.rename("price").dropna()
    price = price[~price.index.duplicated(keep="first")].sort_index()

    return price


def fetch_rf_daily(start="2000-01-01", end="2015-12-31") -> pd.Series:
    """
    Return a daily FF  risk-free series aligned to dates [start, end], as decimal per day.
    """
    data_dict = pdr.DataReader("F-F_Research_Data_Factors_Daily", "famafrench", start=start, end=end)

    rf = data_dict[0].loc[:, "RF"].astype(float) / 100.0
    print(rf)
    rf.index = pd.to_datetime(rf.index)
    rf = rf.sort_index().loc[start:end]
    rf = rf.loc[(rf.index >= pd.to_datetime(start)) & (rf.index <= pd.to_datetime(end))]

    if rf.empty:
        raise RuntimeError("Fama-French RF came back empty for the given dates.")

    return rf.rename("rf")


def build_dataset(
        symbol="SPY",
        start="2000-01-01",
        end="2015-12-31",
        save_csv_path=None,
):
    """
    Produce a tidy daily DataFrame with:
      price, ret (simple), rf (per-day), excess_ret, cumret.
    """
    price = fetch_prices(symbol, start, end)
    rf = fetch_rf_daily(start=start, end=end)

    # align RF to trading days, forward-fill for non-reporting days
    rf = rf.reindex(price.index).ffill()

    ret = price.pct_change()
    df = pd.concat([price, ret.rename("ret"), rf], axis=1).dropna()
    df["excess_ret"] = df["ret"] - df["rf"]
    df["cumret"] = (1.0 + df["ret"]).cumprod()

    if save_csv_path:
        df.to_csv(save_csv_path, index=True)

    return df


if __name__ == "__main__":
    # Example usage; adjust paths as you like.
    df = build_dataset(
        symbol="SPY",
        start="2000-01-01",
        end="2015-12-31",
        save_csv_path="spy_2000_2015.csv",
    )
    print(df.head())
    print(df.tail())
    print(df.describe())
