import numpy as np
import pandas as pd
import yfinance as yf

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
    "get_benchmark_prices",
    "compute_beta",
    "compute_r2",
    "compute_tracking_error",
    "compute_information_ratio",
    "compute_alpha",
    "compute_treynor",
    "compute_up_capture",
    "compute_down_capture",
    "compute_ced_from_mdd_panel",
    "compute_cdd",
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


def get_benchmark_prices(prices: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    idx = prices.index
    start = pd.Timestamp(idx.min()).tz_localize(None)
    end = pd.Timestamp(idx.max()).tz_localize(None) + pd.Timedelta(days=1)

    # explicit auto_adjust to silence future warning; adjusted series resides in "Close"
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # pick adjusted close robustly for single- or multi-index columns
    def _pick_close(frame: pd.DataFrame) -> pd.Series:
        if isinstance(frame.columns, pd.MultiIndex):
            lvl0 = frame.columns.get_level_values(0)
            if "Close" in lvl0:
                sub = frame["Close"]
            elif "Adj Close" in lvl0:
                sub = frame["Adj Close"]
            else:
                raise KeyError(f"Neither 'Close' nor 'Adj Close' present in columns: {list(set(lvl0))}")
            return sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
        else:
            if "Close" in frame.columns:
                return frame["Close"]
            if "Adj Close" in frame.columns:
                return frame["Adj Close"]
            raise KeyError(f"Neither 'Close' nor 'Adj Close' present in columns: {list(frame.columns)}")

    s = _pick_close(df).rename(ticker)
    # IMPORTANT: do NOT reindex/ffill to the asset calendar; keep native trading days
    return s


def _benchmark_returns(prices: pd.DataFrame, ticker: str = "SPY", log: bool = True) -> pd.Series:
    s = get_benchmark_prices(prices, ticker=ticker)
    rb = _returns(s.to_frame(), log=log).iloc[:, 0]
    return rb


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
        risk_free: float = 0.0,  # per-period rf on same sampling as returns
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
        target: float = 0.0,  # per-period target (often 0)
        log: bool = True,
        periods_per_year: int = 252,
        annualize: bool = True,
) -> pd.Series:
    r = _returns(prices, log=log)
    diff = r - target
    neg = diff.clip(upper=0.0)
    downside = np.sqrt((neg ** 2).mean(skipna=True)).replace(0.0, np.nan)
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
    return np.sqrt((dd_pct ** 2).mean(skipna=True))


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
        M = (v[:, None] >= v[None, :])  # pairwise comparisons (>=)
        L = np.tri(n, k=0, dtype=bool)  # restrict to j <= i
        s = (M & L).sum(axis=1) - 1  # s_i
        num = s.sum()  # sum_i s_i
        den = n * (n + 1) / 2.0  # 1+...+n
        out[c] = 100.0 * (num / den)
    return pd.Series(out)


def compute_cii(prices: pd.DataFrame) -> pd.Series:
    """
    CII (%) = 100 - CSI, i.e., cumulative instability (cumulative drawdown) share.
    """
    csi = compute_csi(prices)
    return 100.0 - csi


# ---------- benchmark-relative metrics ----------

def compute_beta(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        log: bool = True,
        min_obs: int = 30,
        var_eps: float = 1e-12,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    Xc = r.sub(r.mean(skipna=True), axis=0)
    yc = rb - rb.mean(skipna=True)

    # pairwise valid counts per asset
    cnt = Xc.notna().mul(yc.notna(), axis=0).sum(axis=0)

    cov = (Xc.mul(yc, axis=0)).mean(skipna=True)  # Series per asset
    var_b = float((yc ** 2).mean(skipna=True))  # scalar

    beta = cov / var_b if (np.isfinite(var_b) and var_b > var_eps) else pd.Series(np.nan, index=r.columns)
    beta[cnt < min_obs] = np.nan
    return beta


def compute_r2(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        log: bool = True,
        min_obs: int = 30,
        var_eps: float = 1e-12,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    Xc = r.sub(r.mean(skipna=True), axis=0)
    yc = rb - rb.mean(skipna=True)

    cnt = Xc.notna().mul(yc.notna(), axis=0).sum(axis=0)

    cov = (Xc.mul(yc, axis=0)).mean(skipna=True)  # Series
    sig_i = np.sqrt((Xc ** 2).mean(skipna=True))  # Series
    sig_b = float(np.sqrt((yc ** 2).mean(skipna=True)))  # scalar

    if not (np.isfinite(sig_b) and sig_b > np.sqrt(var_eps)):
        out = pd.Series(np.nan, index=r.columns)
    else:
        den = sig_i * sig_b
        den = den.replace(0.0, np.nan)
        rho = cov / den
        out = rho ** 2

    out[cnt < min_obs] = np.nan
    return out


def compute_tracking_error(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        log: bool = True,
        annualize: bool = True,
        periods_per_year: int = 252,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    a = r.sub(rb, axis=0)
    te = a.std(ddof=0, skipna=True)
    return te * np.sqrt(periods_per_year) if annualize else te


def compute_information_ratio(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        log: bool = True,
        annualize: bool = True,
        periods_per_year: int = 252,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    a = r.sub(rb, axis=0)
    mu_a = a.mean(skipna=True)
    se_a = a.std(ddof=0, skipna=True).replace(0.0, np.nan)
    ir = mu_a / se_a
    return ir * np.sqrt(periods_per_year) if annualize else ir


def compute_alpha(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        risk_free: float = 0.0,
        log: bool = True,
        annualize: bool = True,
        periods_per_year: int = 252,
        min_obs: int = 30,
        var_eps: float = 1e-12,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    Xc = r.sub(r.mean(skipna=True), axis=0)
    yc = rb - rb.mean(skipna=True)
    cnt = Xc.notna().mul(yc.notna(), axis=0).sum(axis=0)

    cov = (Xc.mul(yc, axis=0)).mean(skipna=True)  # Series
    var_b = float((yc ** 2).mean(skipna=True))  # scalar
    beta = cov / var_b if (np.isfinite(var_b) and var_b > var_eps) else pd.Series(np.nan, index=r.columns)

    ei = r - risk_free
    em = rb - risk_free
    mu_ei = ei.mean(skipna=True)  # Series
    mu_em = float(em.mean(skipna=True))  # scalar

    alpha = mu_ei - beta * mu_em
    alpha[cnt < min_obs] = np.nan
    return alpha * periods_per_year if annualize else alpha


def compute_treynor(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        risk_free: float = 0.0,
        log: bool = True,
        annualize: bool = True,
        periods_per_year: int = 252,
        min_obs: int = 30,
        var_eps: float = 1e-12,
        beta_eps: float = 1e-8,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    Xc = r.sub(r.mean(skipna=True), axis=0)
    yc = rb - rb.mean(skipna=True)
    cnt = Xc.notna().mul(yc.notna(), axis=0).sum(axis=0)

    cov = (Xc.mul(yc, axis=0)).mean(skipna=True)  # Series
    var_b = float((yc ** 2).mean(skipna=True))  # scalar
    beta = cov / var_b if (np.isfinite(var_b) and var_b > var_eps) else pd.Series(np.nan, index=r.columns)

    excess_mean = (r - risk_free).mean(skipna=True)  # Series
    beta = beta.where(beta.abs() > beta_eps)  # treat tiny β as NaN
    trey = excess_mean / beta
    trey[cnt < min_obs] = np.nan
    return trey * periods_per_year if annualize else trey


def compute_up_capture(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        log: bool = True,
        min_days: int = 5,
        denom_eps: float = 1e-10,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    mask = rb > 0
    n = int(mask.sum())
    if n < min_days:
        return pd.Series(np.nan, index=r.columns)

    if log:
        # geometric capture using log-returns
        num = np.exp(r.where(mask).sum(skipna=True)) - 1.0  # Series
        den = float(np.exp(rb.where(mask).sum(skipna=True)) - 1.0)  # scalar
    else:
        num = (1.0 + r.where(mask)).prod(skipna=True) - 1.0
        den = float((1.0 + rb.where(mask)).prod(skipna=True) - 1.0)

    if not np.isfinite(den) or abs(den) < denom_eps:
        return pd.Series(np.nan, index=r.columns)
    return num / den


def compute_down_capture(
        prices: pd.DataFrame,
        bench_prices: pd.Series | None = None,
        bench_ticker: str = "SPY",
        log: bool = True,
        min_days: int = 5,
        denom_eps: float = 1e-10,
) -> pd.Series:
    rb = _benchmark_returns(prices, ticker=bench_ticker, log=log) if bench_prices is None else _returns(
        bench_prices.to_frame(), log=log).iloc[:, 0]
    r = _returns(prices, log=log).align(rb, axis=0, join="inner")[0]
    rb = rb.loc[r.index]

    mask = rb < 0
    n = int(mask.sum())
    if n < min_days:
        return pd.Series(np.nan, index=r.columns)

    if log:
        num = np.exp(r.where(mask).sum(skipna=True)) - 1.0  # Series (negative expected)
        den = float(np.exp(rb.where(mask).sum(skipna=True)) - 1.0)  # scalar (negative)
    else:
        num = (1.0 + r.where(mask)).prod(skipna=True) - 1.0
        den = float((1.0 + rb.where(mask)).prod(skipna=True) - 1.0)

    if not np.isfinite(den) or abs(den) < denom_eps:
        return pd.Series(np.nan, index=r.columns)
    # Down-capture is typically reported as positive (>100% worse than market); sign comes out positive here
    return num / den


def compute_ced_from_mdd_panel(mdd_panel: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    """
    Conditional Expected Drawdown (percent) from a per-window MDD panel.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    out = {}
    for c in mdd_panel.columns:
        x = mdd_panel[c].dropna().to_numpy()
        m = x.size
        if m == 0:
            out[c] = np.nan
            continue
        k = int(np.floor(m * (1.0 - alpha)))
        if k < 1:
            k = 1

        idx = np.argpartition(x, -k)[-k:]
        out[c] = float(x[idx].mean())
    return pd.Series(out)


def compute_cdd(prices: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    """
    Conditional Drawdown at level alpha (CDD_alpha) in % per ticker.
    Definition (Chekhlov–Uryasev–Zabarankin):
      - r_t = simple return (P_t / P_{t-1} - 1).
      - w_k = sum_{l=1..k} r_l  (uncompounded cumulative returns; w_0 = 0).
      - xi_k = max_{0<=j<=k} w_j - w_k  (drawdown series, >= 0).
      - CDD_alpha = tail-mean of {xi_k} at level alpha with exact discrete weighting:
          zeta = inf{s : F_xi(s) >= alpha}  (alpha-quantile, left-continuous),
          pi   = #{xi_k <= zeta}/N,
          CDD  = ((pi - alpha)/(1-alpha))*zeta + (1/((1-alpha)*N)) * sum_{xi_k > zeta} xi_k.
    Returns positive percentage values (xi * 100).
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1).")

    out = {}
    for c in prices.columns:
        p = prices[c].dropna()
        if p.size < 2:
            out[c] = np.nan
            continue

        # simple returns within window
        r = p.pct_change().dropna().to_numpy(dtype=float)
        if r.size == 0:
            out[c] = np.nan
            continue

        # uncompounded cumulative path w_k (w_0=0 is enforced by the max with 0 below)
        w = np.cumsum(r)

        # drawdown series xi_k = max_{j<=k} w_j - w_k, with baseline w_0=0
        m = np.maximum.accumulate(np.maximum(0.0, w))
        xi = m - w  # length N

        N = xi.size
        if N == 0:
            out[c] = np.nan
            continue

        # paper’s alpha-quantile: zeta = x_{(ceil(alpha*N))}, left-continuous empirical CDF
        srt = np.sort(xi)
        k = int(np.ceil(alpha * N))
        k = 1 if k < 1 else k  # guard alpha=0 -> take minimum element
        zeta = float(srt[k - 1])

        # pi(zeta) = #{xi <= zeta}/N (handle float comparisons directly)
        pi = float((xi <= zeta).sum()) / float(N)

        # sum over strictly greater than zeta
        tail_sum = float(xi[xi > zeta].sum())

        cdd = ((pi - alpha) / (1.0 - alpha)) * zeta + (tail_sum / ((1.0 - alpha) * N))
        out[c] = 100.0 * cdd  # percent

    return pd.Series(out, index=prices.columns, dtype=float)
