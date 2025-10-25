import numpy as np
import pandas as pd

__all__ = [
    "compute_return",
    "compute_volatility",
    "compute_sharpe",
    "compute_sortino",
    "compute_var",
    "compute_es",
    "compute_mdd",
    "compute_ulcer_index",
    "compute_avg_drawdown",
    "compute_max_dd_duration",
    "compute_avg_dd_duration",
    "compute_calmar",
]

# ---------- helpers ----------

def _returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    # per-period returns from prices (aligns with rolling window math)
    return np.log(prices / prices.shift(1)) if log else (prices / prices.shift(1) - 1)

def _drawdown_fraction(prices: pd.DataFrame) -> pd.DataFrame:
    # drawdown as fraction (<= 0): P_t / max_{s<=t} P_s - 1
    peak = prices.cummax()
    return prices.divide(peak) - 1.0

def _streak_lengths(mask: pd.Series) -> np.ndarray:
    # lengths of contiguous True runs in a boolean Series; NaN treated as False
    x = mask.fillna(False).to_numpy(dtype=bool)
    if x.size == 0:
        return np.array([], dtype=int)
    # indices where a run starts: True and previous is False
    starts = np.flatnonzero(x & np.concatenate([[True], ~x[:-1]]))
    # indices where a run ends: True and next is False (inclusive)
    ends = np.flatnonzero(x & np.concatenate([~x[1:], [True]]))
    return (ends - starts + 1).astype(int)

# ---------- return / risk metrics ----------

def compute_return(prices: pd.DataFrame, log: bool = True) -> pd.Series:
    r = _returns(prices, log=log)
    # window cumulative: sum of log-returns (log) or P_T/P_0 - 1 (simple)
    return r.sum(skipna=True) if log else (prices.iloc[-1] / prices.iloc[0] - 1.0)

def compute_volatility(
    prices: pd.DataFrame,
    log: bool = True,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    r = _returns(prices, log=log)
    sigma = r.std(ddof=0, skipna=True)
    return sigma * np.sqrt(periods_per_year) if annualize else sigma

def compute_sharpe(
    prices: pd.DataFrame,
    risk_free: float = 0.0,         # per-period rf on same sampling as returns
    log: bool = True,
    periods_per_year: int = 252,
    annualize: bool = True,
) -> pd.Series:
    r = _returns(prices, log=log)
    excess = r - risk_free
    mu = excess.mean(skipna=True)
    sigma = r.std(ddof=0, skipna=True).replace(0.0, np.nan)
    s = mu / sigma
    return s * np.sqrt(periods_per_year) if annualize else s

def compute_sortino(
    prices: pd.DataFrame,
    target: float = 0.0,            # per-period target (often 0)
    log: bool = True,
    periods_per_year: int = 252,
    annualize: bool = True,
) -> pd.Series:
    r = _returns(prices, log=log)
    diff = r - target
    neg = diff.clip(upper=0.0)
    downside = np.sqrt((neg**2).mean(skipna=True)).replace(0.0, np.nan)
    mu_excess = diff.mean(skipna=True)
    s = mu_excess / downside
    return s * np.sqrt(periods_per_year) if annualize else s

def compute_var(prices: pd.DataFrame, alpha: float = 0.95, log: bool = True) -> pd.Series:
    r = _returns(prices, log=log)
    q = r.quantile(1.0 - alpha, interpolation="linear")
    return -q  # positive loss magnitude

def compute_es(prices: pd.DataFrame, alpha: float = 0.95, log: bool = True) -> pd.Series:
    r = _returns(prices, log=log)
    q = r.quantile(1.0 - alpha, interpolation="linear")
    tail = r[r.le(q)]
    return -tail.mean(skipna=True)  # positive loss magnitude

# ---------- drawdown family (price-path metrics) ----------

def compute_mdd(prices: pd.DataFrame) -> pd.Series:
    """
    Maximum Drawdown (percentage, positive magnitude).
    DD_t = P_t / cummax(P)_t - 1  (<= 0).  MDD = -min(DD_t) * 100.
    """
    dd = _drawdown_fraction(prices)
    return (-dd.min(skipna=True)) * 100.0

def compute_ulcer_index(prices: pd.DataFrame) -> pd.Series:
    """
    Ulcer Index (percent): sqrt( mean( DD_t(%)^2 ) ), using all days in window.
    DD_t(%) = 100 * (P_t / cummax(P)_t - 1), which is <= 0.
    """
    dd_pct = 100.0 * _drawdown_fraction(prices)
    return np.sqrt((dd_pct**2).mean(skipna=True))

def compute_avg_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Average Drawdown (percent, over drawdown days only).
    = mean( -DD_t * 100 ) for days with DD_t < 0.
    """
    dd = _drawdown_fraction(prices)
    dd_mag_pct = (-dd.where(dd < 0)) * 100.0
    return dd_mag_pct.mean(skipna=True).fillna(0.0)

def compute_max_dd_duration(prices: pd.DataFrame) -> pd.Series:
    """
    Longest run (in days) spent below the running peak inside the window.
    """
    dd = _drawdown_fraction(prices)
    mask = dd < 0
    # apply column-wise to get max streak
    return mask.apply(lambda s: _streak_lengths(s).max() if s.notna().any() else 0, axis=0).astype(float)

def compute_avg_dd_duration(prices: pd.DataFrame) -> pd.Series:
    """
    Mean length (in days) of drawdown episodes inside the window.
    """
    dd = _drawdown_fraction(prices)
    mask = dd < 0
    def _avg_len(s: pd.Series) -> float:
        runs = _streak_lengths(s)
        return runs.mean() if runs.size else 0.0
    return mask.apply(_avg_len, axis=0).astype(float)

def compute_calmar(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """
    Calmar ratio = annualized simple return / |MDD|,
    where MDD is the percentage drawdown magnitude.
    Annualization uses geometric scaling from window length.
    """
    # cumulative simple return over the window
    cum_simple = (prices.iloc[-1] / prices.iloc[0]) - 1.0
    T = max(len(prices) - 1, 1)  # number of periods in the window
    ann_simple = (1.0 + cum_simple) ** (periods_per_year / T) - 1.0

    mdd_pct = compute_mdd(prices)  # positive percent
    mdd_frac = (mdd_pct / 100.0).replace(0.0, np.nan)
    return ann_simple / mdd_frac


# ---------- stability family (CSI / CII) ----------

def compute_csi(prices: pd.DataFrame) -> pd.Series:
    """
    CSI (%) per ticker, following the paper's definition:
      For each time i (1..n), s_i = #{ j <= i : y_i >= y_j } - 1.
      Reference sum = 1 + 2 + ... + n.
      CSI = 100 * ( sum_i s_i ) / ( sum_{i=1}^n i ).
    Ties count as stable (>=). NaNs are dropped per column (window-wise).
    """
    out = {}
    for c in prices.columns:
        y = prices[c].dropna()
        n = len(y)
        if n < 2:
            out[c] = np.nan
            continue
        v = y.to_numpy()
        M = (v[:, None] >= v[None, :])            # pairwise comparisons (>=)
        L = np.tri(n, k=0, dtype=bool)            # restrict to j <= i
        s = (M & L).sum(axis=1) - 1               # s_i
        num = s.sum()                              # sum_i s_i
        den = n * (n + 1) / 2.0                    # 1+...+n
        out[c] = 100.0 * (num / den)
    return pd.Series(out)

def compute_cii(prices: pd.DataFrame) -> pd.Series:
    """
    CII (%) = 100 - CSI, i.e., cumulative instability (cumulative drawdown) share.
    """
    csi = compute_csi(prices)
    return 100.0 - csi