# metrics.py
import numpy as np
import pandas as pd

__all__ = [
    # wrappers
    "to_prices_df",
    "metric_table",
    "turnover_cost_table",
    # core metrics (copied from calculate_metrics.py where applicable)
    "compute_total_return",
    "compute_cagr",
    "compute_volatility",
    "compute_sharpe",
    "compute_sortino",
    "compute_mdd",
    "compute_ulcer_index",
    "compute_calmar",
    "compute_csi",
    "compute_cii",
    # window metrics (Phase 0 cluster representatives)
    "compute_window_ulcer",
    "compute_window_avg_dd_duration",
    "compute_window_mdd_demeaned",
    "compute_window_es",
]

# ---------- helpers (copied) ----------

def _returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    # per-period returns from prices (aligns with rolling window math)
    return np.log(prices / prices.shift(1)) if log else (prices / prices.shift(1) - 1)


def _drawdown_fraction(prices: pd.DataFrame) -> pd.DataFrame:
    # drawdown as fraction (<= 0): P_t / max_{s<=t} P_s - 1
    peak = prices.cummax()
    return prices.divide(peak) - 1.0


# ---------- input convenience (new) ----------

def to_prices_df(x, name="portfolio") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame(name)
    x = np.asarray(x).reshape(-1)
    return pd.DataFrame({name: x})


# ---------- metrics required by the plan ----------

def compute_total_return(prices: pd.DataFrame) -> pd.Series:
    prices = to_prices_df(prices)
    return (prices.iloc[-1] / prices.iloc[0]) - 1.0


def compute_cagr(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    prices = to_prices_df(prices)
    T = max(len(prices) - 1, 1)  # number of periods in the window
    return (prices.iloc[-1] / prices.iloc[0]) ** (periods_per_year / T) - 1.0


def compute_volatility(
        prices: pd.DataFrame,
        log: bool = True,
        annualize: bool = True,
        periods_per_year: int = 252,
) -> pd.Series:
    # copied from calculate_metrics.py (same logic)
    prices = to_prices_df(prices)
    r = _returns(prices, log=log)
    sigma = r.std(ddof=0, skipna=True)
    return sigma * np.sqrt(periods_per_year) if annualize else sigma


def compute_sharpe(
        prices: pd.DataFrame,
        risk_free: float = 0.0,  # per-period rf on same sampling as returns
        log: bool = True,
        periods_per_year: int = 252,
        annualize: bool = True,
) -> pd.Series:
    # copied from calculate_metrics.py (same logic)
    prices = to_prices_df(prices)
    r = _returns(prices, log=log)
    excess = r - risk_free
    mu = excess.mean(skipna=True)
    sigma = r.std(ddof=0, skipna=True).replace(0.0, np.nan)
    s = mu / sigma
    return s * np.sqrt(periods_per_year) if annualize else s


def compute_sortino(
        prices: pd.DataFrame,
        target: float = 0.0,  # per-period target (often 0)
        log: bool = True,
        periods_per_year: int = 252,
        annualize: bool = True,
) -> pd.Series:
    # copied from calculate_metrics.py (same logic)
    prices = to_prices_df(prices)
    r = _returns(prices, log=log)
    diff = r - target
    neg = diff.clip(upper=0.0)
    downside = np.sqrt((neg ** 2).mean(skipna=True)).replace(0.0, np.nan)
    mu_excess = diff.mean(skipna=True)
    s = mu_excess / downside
    return s * np.sqrt(periods_per_year) if annualize else s


def compute_mdd(prices: pd.DataFrame) -> pd.Series:
    # copied from calculate_metrics.py (same logic)
    prices = to_prices_df(prices)
    dd = _drawdown_fraction(prices)
    return (-dd.min(skipna=True)) * 100.0


def compute_ulcer_index(prices: pd.DataFrame) -> pd.Series:
    # copied from calculate_metrics.py (same logic)
    prices = to_prices_df(prices)
    dd_pct = 100.0 * _drawdown_fraction(prices)
    return np.sqrt((dd_pct ** 2).mean(skipna=True))


def compute_calmar(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    # copied from calculate_metrics.py (same logic)
    prices = to_prices_df(prices)
    cum_simple = (prices.iloc[-1] / prices.iloc[0]) - 1.0
    T = max(len(prices) - 1, 1)
    ann_simple = (1.0 + cum_simple) ** (periods_per_year / T) - 1.0

    mdd_pct = compute_mdd(prices)
    mdd_frac = (mdd_pct / 100.0).replace(0.0, np.nan)
    return ann_simple / mdd_frac


def compute_csi(prices: pd.DataFrame) -> pd.Series:
    """
    CSI computed with denominator n*(n-1)/2 where n is the number of input values.

    With s_i = sum_{j=0..i-1} a_ij and a_i0 = 0 by definition (no y_0),
    the maximum achievable sum across all i is 0+1+2+...+(n-1) = n(n-1)/2.

    Note: prior CSI specification documents (Iashchenko docx, DD_evaluation.pdf)
    state n(n+1)/2, which is incorrect — that denominator yields CSI < 100% for
    perfectly ascending series. The convention used here is the mathematically
    consistent one: a perfectly ascending series of length n returns CSI = 100%.
    """
    # copied from calculate_metrics.py (same logic; naive O(n^2))
    prices = to_prices_df(prices)
    out = {}
    for c in prices.columns:
        y = prices[c].dropna()
        n = len(y)
        if n < 2:
            out[c] = np.nan
            continue
        v = y.to_numpy()
        M = (v[:, None] >= v[None, :])  # pairwise comparisons (>=)
        L = np.tri(n, k=0, dtype=bool)  # restrict to j <= i
        s = (M & L).sum(axis=1) - 1
        num = s.sum()
        den = n * (n - 1) / 2.0
        out[c] = 100.0 * (num / den)
    return pd.Series(out)


def compute_cii(prices: pd.DataFrame) -> pd.Series:
    # copied from calculate_metrics.py
    csi = compute_csi(prices)
    return 100.0 - csi


# ---------- window metrics (cluster representatives, Phase 0) ----------

def compute_window_ulcer(series) -> float:
    """Cluster 2 representative — drawdown magnitude (RMS form).

    Ulcer Index over window: sqrt(mean(dd^2)) computed on the running-max
    drawdowns of the series. Returned as fraction (>= 0).

    Used in the Martin ratio (return / Ulcer). Chosen over MDD because the
    per-step Δ of MDD is sparse — it changes only when a new low forms.
    Ulcer averages over the window and provides a dense, smooth signal.
    """
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    if s.size < 2:
        return 0.0
    peak = np.maximum.accumulate(s)
    dd = s / peak - 1.0  # <= 0
    return float(np.sqrt(np.mean(dd ** 2)))


def compute_window_avg_dd_duration(series) -> float:
    """Cluster 1 representative — drawdown duration.

    Mean length of drawdown episodes within the window, in steps.
    A drawdown episode is a maximal contiguous run of steps where
    series < running_max. Returns 0.0 if no drawdown episodes occur.
    """
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    if s.size < 2:
        return 0.0
    peak = np.maximum.accumulate(s)
    underwater = s < peak
    durations = []
    run = 0
    for u in underwater:
        if u:
            run += 1
        else:
            if run > 0:
                durations.append(run)
            run = 0
    if run > 0:
        durations.append(run)
    if not durations:
        return 0.0
    return float(np.mean(durations))


def compute_window_mdd_demeaned(series) -> float:
    """Cluster 4 representative — demeaned drawdown.

    MDD computed on the price path reconstructed from demeaned log returns.
    This isolates within-trend volatility from drift: a series with strong
    upward drift can have MDD = 0 (no breach of running max) while still
    exhibiting significant short-term reversals; demeaning surfaces those.

    Procedure:
      1. log_returns = log(s[1:] / s[:-1])
      2. demeaned = log_returns - mean(log_returns)
      3. reconstructed = cumprod(exp(demeaned)) prefixed with 1.0
      4. return MDD on the reconstructed path
    """
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    if s.size < 3:
        return 0.0
    log_ret = np.log(s[1:] / s[:-1])
    demeaned = log_ret - np.mean(log_ret)
    reconstructed = np.concatenate([[1.0], np.cumprod(np.exp(demeaned))])
    peak = np.maximum.accumulate(reconstructed)
    dd = reconstructed / peak - 1.0
    return float(-dd.min())


def compute_window_es(series, alpha: float = 0.05) -> float:
    """Cluster 0 representative — tail risk (Expected Shortfall / CVaR).

    Expected Shortfall at confidence (1 - alpha): mean of the worst alpha-fraction
    of log returns within the window. Returned as a positive number representing
    loss magnitude (so a more-negative tail produces a larger ES value).

    Convention: at alpha=0.05 with W=60 log-returns (so 59 returns), the worst
    ceil(0.05 * 59) = 3 returns are averaged. Minimum 1 sample taken.

    Chosen over Std because Std loses tail-shape information, and over VaR
    because VaR is non-convex and not subadditive (i.e. it can discourage
    diversification, which is exactly the wrong incentive for portfolio RL).
    """
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    if s.size < 2:
        return 0.0
    log_ret = np.log(s[1:] / s[:-1])
    n = log_ret.size
    k = max(1, int(np.ceil(alpha * n)))
    worst = np.sort(log_ret)[:k]  # most-negative returns
    return float(-np.mean(worst))


# ---------- tables used in eval (new, small) ----------

def metric_table(prices, periods_per_year: int = 252, log: bool = True, excess_prices=None) -> pd.DataFrame:
    P = to_prices_df(prices)
    out = pd.DataFrame(index=P.columns)
    out["total_return"] = compute_total_return(P)
    out["cagr"] = compute_cagr(P, periods_per_year=periods_per_year)
    out["vol_ann"] = compute_volatility(P, log=log, annualize=True, periods_per_year=periods_per_year)
    out["sharpe_ann"] = compute_sharpe(P, log=log, annualize=True, periods_per_year=periods_per_year)
    out["sortino_ann"] = compute_sortino(P, log=log, annualize=True, periods_per_year=periods_per_year)
    out["mdd_pct"] = compute_mdd(P)
    out["calmar"] = compute_calmar(P, periods_per_year=periods_per_year)
    out["ulcer"] = compute_ulcer_index(P)
    csi_input = to_prices_df(excess_prices, name=P.columns[0]) if excess_prices is not None else P
    out["csi"] = compute_csi(csi_input)
    out["cii"] = compute_cii(csi_input)
    return out


def turnover_cost_table(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects env history dataframe with columns:
      - 'turnover' (L1 turnover per step)
      - 'cost' (the fractional haircut coefficient used in env: cost = c * turnover)
    """
    h = history_df
    to = np.asarray(h["turnover"], dtype=float) if "turnover" in h.columns else np.array([])
    c = np.asarray(h["cost"], dtype=float) if "cost" in h.columns else np.array([])

    total_turnover = float(np.nansum(to)) if to.size else np.nan
    avg_turnover = float(np.nanmean(to)) if to.size else np.nan
    total_cost = float(np.nansum(c)) if c.size else np.nan
    avg_cost = float(np.nanmean(c)) if c.size else np.nan

    # cumulative multiplicative impact from the env's (1 - cost_t) haircut
    cost_multiplier = float(np.nanprod(1.0 - c)) if c.size else np.nan
    cost_impact = float(1.0 - cost_multiplier) if np.isfinite(cost_multiplier) else np.nan

    return pd.DataFrame(
        {
            "total_turnover": [total_turnover],
            "avg_turnover": [avg_turnover],
            "total_cost_coeff_sum": [total_cost],
            "avg_cost_coeff": [avg_cost],
            "cost_multiplier_prod": [cost_multiplier],
            "cost_impact": [cost_impact],
        }
    )
