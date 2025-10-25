'''import pandas as pd

# read data fram csv
df = pd.DataFrame(pd.read_csv('data/company_prices.csv', header=2).iloc[:, 1:], index=range(252)).rename(
    columns={'Ticker': 'Date'})

print(df.iloc[-5:, :5])
# turn prices to float
df[df.columns.difference(['Date'])] = (
    df[df.columns.difference(['Date'])].replace(',', '.', regex=True).astype('float64')
)

# calculate mean and stdev values
data = pd.DataFrame({'mean': [df[col].mean() for col in df.columns],
                     'stdev': [df[col].std() for col in df.columns]},
                    index=df.columns)

if __name__ == "__main__":
    # output mean and stdev data
    print(data[:5])
'''


import pandas as pd

def load_prices(csv_path):
    raw = pd.read_csv(csv_path, header=None)

    # Header rows
    sectors = raw.iloc[0, 1:].astype(str).tolist()
    names   = raw.iloc[1, 1:].astype(str).tolist()
    tickers = raw.iloc[2, 1:].astype(str).tolist()

    # Data begins on row 3; first column is Date (row 2 says "Ticker" -> "Date")
    df = raw.iloc[3:].copy()
    df.columns = ["Date"] + tickers

    # Parse Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset=["Date"])
          .set_index("Date")
    )

    # Convert decimal comma strings like "129,41" -> 129.41
    # also strip spaces / non-breaking spaces
    for c in df.columns:
        s = (
            df[c].astype(str)
                 .str.replace("\u00a0", "", regex=False)  # NBSP
                 .str.replace(" ", "", regex=False)
                 .str.replace(",", ".", regex=False)
        )
        df[c] = pd.to_numeric(s, errors="coerce")

    # Drop columns that are entirely NaN; enforce float64
    df = df.loc[:, ~df.isna().all()].astype("float64")

    # Build simple metadata table (ticker -> name, sector)
    meta = pd.DataFrame(
        {"ticker": tickers, "name": names, "sector": sectors}
    ).set_index("ticker")

    return df, meta


def validate_prices(prices, allow_negative=False, min_non_na_frac=0.8):
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Index must be increasing by date.")
    if (1.0 - prices.isna().mean() < min_non_na_frac).any():
        bad = list((1.0 - prices.isna().mean())[lambda s: s < min_non_na_frac].index)
        raise ValueError(f"Too many NaNs in columns: {bad}")
    if not allow_negative and (prices.min(numeric_only=True).min() < 0):
        raise ValueError("Found negative prices; check your input.")
    return True
