# indicators.py
# Leaf module: technical-indicator series used by the env's default_feature_fn.
# Pure numpy, no project dependencies.
import numpy as np


def _ema(x, period):
    """EMA bootstrapped from SMA at index period-1. Returns NaN before period-1."""
    alpha = 2.0 / (period + 1)
    n = len(x)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(x[:period])
    for i in range(period, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi_series(close, period=14):
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(close)
    up   = np.where(delta > 0,  delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    for i in range(period, n):
        au = np.mean(up[i - period:i])
        ad = np.mean(down[i - period:i])
        out[i] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return out


def macd_hist_series(close, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow                  # NaN until slow-1

    n = len(macd)
    signal_line = np.full(n, np.nan)
    first_valid = slow - 1                      # first non-NaN index of macd
    bootstrap_idx = first_valid + signal - 1    # need 'signal' macd values to bootstrap
    if bootstrap_idx >= n:
        return np.full(n, np.nan)

    signal_line[bootstrap_idx] = np.mean(macd[first_valid:first_valid + signal])
    alpha = 2.0 / (signal + 1)
    for i in range(bootstrap_idx + 1, n):
        signal_line[i] = alpha * macd[i] + (1.0 - alpha) * signal_line[i - 1]

    return macd - signal_line                   # NaN until slow + signal - 2


def bb_z_series(close, period=20):
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n + 1):
        win = close[i - period:i]
        m = np.mean(win); s = np.std(win)
        out[i - 1] = 0.0 if s == 0 else (close[i - 1] - m) / s
    return out


def vol_series(log_returns, period=20):
    n = len(log_returns)
    out = np.full(n, np.nan)
    for i in range(period, n + 1):
        out[i - 1] = np.std(log_returns[i - period:i])
    return out


def momentum_series(close, period=20):
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        c0 = close[i - period]
        out[i] = (close[i] / c0 - 1.0) if c0 > 0 else 0.0
    return out


def volume_ratio_series(volume, period=20):
    n = len(volume)
    out = np.full(n, np.nan)
    for i in range(period, n + 1):
        m = np.mean(volume[i - period:i])
        out[i - 1] = (volume[i - 1] / m) if m > 0 else 1.0
    return out


def z_score_last(series):
    """Z-score the last valid value against the rolling distribution of itself."""
    s = np.asarray(series, dtype=np.float64)
    valid = s[~np.isnan(s)]
    if valid.size < 2:
        return 0.0
    mu = np.mean(valid); sd = np.std(valid)
    if sd == 0:
        return 0.0
    return float((valid[-1] - mu) / sd)


def default_feature_fn(close_win, high_win, low_win, vol_win):
    """
    Inputs: four (W, N) arrays. W is the env window (60), N is asset count.
    Output: flat (6*N,) feature vector. Per-asset features in fixed order:
      [RSI14, MACD_hist, BB_z20, vol20, mom20, vol_ratio20]
    Each feature is z-scored against the rolling distribution of itself
    over the same W-window (matches plan.md §2.3).
    high_win and low_win are accepted for signature stability but unused
    by the current indicator set; kept so future indicators (ATR, Stoch)
    drop in without changing the env contract.
    """
    W, N = close_win.shape
    out = np.zeros((N, 6), dtype=np.float64)
    for j in range(N):
        c = close_win[:, j]
        v = vol_win[:, j]
        log_ret = np.diff(np.log(np.maximum(c, 1e-12)))
        out[j, 0] = z_score_last(rsi_series(c, 14))
        out[j, 1] = z_score_last(macd_hist_series(c, 12, 26, 9))
        out[j, 2] = z_score_last(bb_z_series(c, 20))
        out[j, 3] = z_score_last(vol_series(log_ret, 20))
        out[j, 4] = z_score_last(momentum_series(c, 20))
        out[j, 5] = z_score_last(volume_ratio_series(v, 20))
    return out.reshape(-1)
