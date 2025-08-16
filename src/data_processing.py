import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


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
    Return a daily risk-free series from FRED aligned to dates [start, end], as decimal per day.

    We pull an annualized yield (in percent) and convert to per-day decimal by dividing by 252*100,
    to preserve the script's existing convention where annualization ≈ rf * 252 * 100.
    Priority: DGS1MO → DGS3MO → DTB3 → Yahoo ^IRX.
    """
    # Try FRED constant-maturity 1M
    rf = None
    try:
        s = pdr.DataReader("DGS1MO", "fred", start=start, end=end).squeeze("columns")
        if s is not None and not s.empty:
            rf = s.rename("DGS1MO")
    except Exception:
        rf = None

    # Fallback to 3M constant-maturity
    if rf is None or rf.empty:
        try:
            s = pdr.DataReader("DGS3MO", "fred", start=start, end=end).squeeze("columns")
            if s is not None and not s.empty:
                rf = s.rename("DGS3MO")
        except Exception:
            pass

    # Fallback to 3M T-bill secondary market rate (discount basis, still in percent)
    if rf is None or rf.empty:
        try:
            s = pdr.DataReader("DTB3", "fred", start=start, end=end).squeeze("columns")
            if s is not None and not s.empty:
                rf = s.rename("DTB3")
        except Exception:
            pass

    # Final fallback: Yahoo ^IRX (CBOE 13-week T-bill rate, percent)
    if rf is None or rf.empty:
        try:
            hist = yf.Ticker("^IRX").history(start=start, end=end, auto_adjust=False)
            if hist is not None and not hist.empty:
                rf = hist["Close"].rename("^IRX")
        except Exception:
            pass

    if rf is None or rf.empty:
        raise RuntimeError("Could not retrieve risk-free series from FRED/Yahoo for the given dates.")

    # Convert to per-day decimal to match the rest of the pipeline:
    # percent -> decimal -> divide by 252 to get 'per trading day'
    rf = (rf.astype(float) / 100.0) / 252.0

    # Index handling and clipping (keep the same behavior as before)
    rf.index = pd.to_datetime(rf.index)
    rf = rf.sort_index().loc[start:end]
    rf = rf.loc[(rf.index >= pd.to_datetime(start)) & (rf.index <= pd.to_datetime(end))]

    if rf.empty:
        raise RuntimeError("Risk-free series is empty after clipping to the requested dates.")

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


def plot_spy_and_rf(df: pd.DataFrame,
                    title: str | None = None,
                    use_seaborn: bool = True,
                    annualize_rf: bool = True,
                    outfile: str | None = None,
                    show: bool = True):
    """
    Plot SPY adjusted close vs. risk-free rate over time.

    - df: DataFrame from build_dataset() with columns ['price','rf',...]
    - annualize_rf: if True, show RF as annualized percent (≈ rf * 252 * 100)
    - outfile: optional path to save the figure
    """
    if use_seaborn:
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")  # modern global styling
        except Exception:
            pass  # fall back to plain matplotlib if seaborn missing

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # SPY price (left axis)
    ax1.plot(df.index, df["price"], label="SPY (adj close)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")

    # RF (right axis)
    ax2 = ax1.twinx()
    if annualize_rf:
        rf_plot = df["rf"] * 252 * 100.0
        rf_label = "Risk-free (annualized, %)"
        ax2.set_ylabel(rf_label)
    else:
        rf_plot = df["rf"]
        rf_label = "Risk-free (per day)"
        ax2.set_ylabel(rf_label)
    ax2.plot(df.index, rf_plot, linestyle="--", alpha=0.8, label=rf_label)

    # Date ticks that adapt to the window length
    locator = mdates.AutoDateLocator()
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

    # Title + legend
    title = title or f"SPY vs Risk-free ({df.index.min().date()} → {df.index.max().date()})"
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
    print(df['ret'].dropna().idxmax().strftime('%Y-%m-%d'))
    plot_spy_and_rf(df, title="SPY & Risk-free (FRED)", outfile="spy_rf.png", show=True)
