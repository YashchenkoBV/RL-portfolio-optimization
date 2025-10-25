import pandas as pd


def rolling_windows(df, window, step=1, min_valid_frac=0.9, drop_constant=True):
    """
    Yield (window_end_timestamp, window_df) over df.
    - drops columns with NaN fraction > (1 - min_valid_frac) inside the window
    - optionally drops constant columns inside the window
    """
    n = len(df)
    if window <= 1:
        raise ValueError("window must be > 1")
    if step < 1:
        raise ValueError("step must be >= 1")

    for start in range(0, n - window + 1, step):
        wdf = df.iloc[start:start + window]

        keep = (1.0 - wdf.isna().mean()) >= min_valid_frac
        wdf = wdf.loc[:, keep]

        if drop_constant and wdf.shape[1] > 0:
            varying = wdf.apply(lambda s: s.dropna().nunique() > 1, axis=0)
            wdf = wdf.loc[:, varying]

        if wdf.shape[1] == 0:
            continue

        yield (wdf.index[-1], wdf)

def rolling_apply(df, func, window, step=1, **kwargs):
    """
    Apply func(window_df) over rolling windows and return a tidy DataFrame.
    func should return a pandas Series or a dict (per-window metrics).
    Rows are indexed by window end timestamp.
    """
    rows, idx = [], []
    for end, wdf in rolling_windows(df, window, step, **kwargs):
        res = func(wdf)
        if isinstance(res, dict):
            res = pd.Series(res)
        if not isinstance(res, pd.Series):
            raise TypeError("func must return a pandas Series or a dict")
        rows.append(res)
        idx.append(end)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.index = pd.DatetimeIndex(idx, name="window_end")
    return out
