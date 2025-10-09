from __future__ import annotations

from typing import Iterator, Dict
import pandas as pd


def rolling_windows(
        df: pd.DataFrame,
        window_months: int = 12,
        step_months: int = 1,
        min_coverage: float = 0.95,
) -> Iterator[Dict]:
    """
    Yield month-aligned rolling windows over daily data.
    """

    if "ret" not in df.columns:
        raise ValueError("DataFrame must include 'ret' (daily simple returns).")

    if df.index.tz is not None:
        df = df.tz_localize(None)

    # Full month range covered by the data
    months = pd.period_range(df.index.min().to_period("M"),
                             df.index.max().to_period("M"),
                             freq="M")

    last_start_idx = len(months) - window_months
    if last_start_idx < 0:
        return  # nothing to yield

    # Precompute each month's first/last trading day present in df
    # (using df itself as the trading calendar)
    by_month = df.groupby(df.index.to_period("M"))
    month_first = by_month.head(1).index.to_series().groupby(level=0).first()
    month_last = by_month.tail(1).index.to_series().groupby(level=0).last()

    for i in range(0, last_start_idx + 1, step_months):
        start_per = months[i]
        end_per = months[i + window_months - 1]

        # Determine window boundaries on actual trading days in df
        try:
            start_ts = month_first.loc[start_per]
            end_ts = month_last.loc[end_per]
        except KeyError:
            # Missing a boundary month in the actual data -> skip
            continue

        w = df.loc[start_ts:end_ts]

        # Coverage on 'ret' inside the window
        n = len(w["ret"])
        if n == 0:
            continue
        n_valid = int(w["ret"].notna().sum())
        coverage = n_valid / n
        if coverage < min_coverage or n_valid == 0:
            continue

        # In-window equity curve (start at 1.0)
        ret = w["ret"].fillna(0.0)
        equity = (1.0 + ret).cumprod()
        w = w.copy()
        w["equity"] = equity

        label = f"{start_per.start_time.strftime('%Y-%m')} \u2192 {end_per.start_time.strftime('%Y-%m')}"
        yield {
            "label": label,
            "start": start_ts,
            "end": end_ts,
            "coverage": coverage,
            "data": w,
        }


if __name__ == "__main__":
    pass
