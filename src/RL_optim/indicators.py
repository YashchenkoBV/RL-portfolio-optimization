# indicators.py
# Leaf module: technical-indicator series used by the env's default_feature_fn.
# Pure numpy, no project dependencies.
import json
import numpy as np


# ----------------------------- raw indicator series -----------------------------

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
    """NaN-safe signal-line EMA: bootstraps from valid MACD values only,
    avoiding the np.nan_to_num=0 bias of the previous implementation."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow                  # NaN until slow-1

    n = len(macd)
    signal_line = np.full(n, np.nan)
    first_valid = slow - 1
    bootstrap_idx = first_valid + signal - 1
    if bootstrap_idx >= n:
        return np.full(n, np.nan)

    signal_line[bootstrap_idx] = np.mean(macd[first_valid:first_valid + signal])
    alpha = 2.0 / (signal + 1)
    for i in range(bootstrap_idx + 1, n):
        signal_line[i] = alpha * macd[i] + (1.0 - alpha) * signal_line[i - 1]

    return macd - signal_line


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


# ------------------------------- raw last-value -------------------------------
# Returns the last valid (non-NaN) value of an indicator series, or NaN if none
# exists. This is the unnormalized indicator value at the current step, in its
# native units (RSI in [0,100], MACD-hist in price-unit, BB-z in σ-units, etc.).

def _last_valid(series):
    s = np.asarray(series, dtype=np.float64)
    valid = s[~np.isnan(s)]
    if valid.size == 0:
        return np.nan
    return float(valid[-1])


# --------------------------- raw feature extraction ---------------------------
# Order is canonical and must match calibration / normalization downstream.
FEATURE_NAMES = ("rsi14", "macd_hist", "bb_z20", "vol20", "mom20", "vol_ratio20")
N_FEATURES = len(FEATURE_NAMES)


def raw_feature_fn(close_win, high_win, low_win, vol_win):
    """Compute raw (un-normalized) indicator values at the most recent step
    for each asset.

    Inputs: four (W, N) arrays. W is the env window (>= 35 for MACD validity).
    Output: (N, 6) array of raw indicator values. NaN propagates if any
    indicator's series has no valid values within the input window.

    Used both at training time (inside default_feature_fn) and at calibration
    time (sweeping the train split to fit normalization stats).
    """
    W, N = close_win.shape
    out = np.zeros((N, N_FEATURES), dtype=np.float64)
    for j in range(N):
        c = close_win[:, j]
        v = vol_win[:, j]
        log_ret = np.diff(np.log(np.maximum(c, 1e-12)))
        out[j, 0] = _last_valid(rsi_series(c, 14))
        out[j, 1] = _last_valid(macd_hist_series(c, 12, 26, 9))
        out[j, 2] = _last_valid(bb_z_series(c, 20))
        out[j, 3] = _last_valid(vol_series(log_ret, 20))
        out[j, 4] = _last_valid(momentum_series(c, 20))
        out[j, 5] = _last_valid(volume_ratio_series(v, 20))
    return out


# ----------------------------- normalization stats -----------------------------
# Stats are stored as a JSON dict with two arrays per feature: pooled-across-
# assets mean and std. Pooling across assets is deliberate — see the explanation
# below. Stats are fit ONCE on the train split, then applied unchanged at val
# and test time. This is the train-set z-score convention, which is the standard
# preprocessing step in ML pipelines.

class FeatureNormalizer:
    """Train-set z-score normalizer for indicator features.

    Fit pooled-across-assets stats per feature on a price tensor, then apply
    the same (mu, sigma) to any future window, regardless of asset or split.

    A single (mu, sigma) per feature — not per (feature, asset) — is the right
    pooling: it preserves cross-asset relative information (asset A's RSI is
    higher than asset B's RSI today => z(A,RSI) > z(B,RSI)) while still
    standardizing the scale of each feature across the training distribution.
    """

    def __init__(self, mu=None, sigma=None):
        # mu, sigma: arrays of shape (N_FEATURES,) or None (uninitialized).
        self.mu = None if mu is None else np.asarray(mu, dtype=np.float64)
        self.sigma = None if sigma is None else np.asarray(sigma, dtype=np.float64)

    @property
    def is_fit(self):
        return self.mu is not None and self.sigma is not None

    def fit_from_price_tensor(self, P, H, L, V, window=60, stride=1):
        """Sweep windows of length `window` across the train price tensor,
        compute raw feature values at each step, accumulate, then take pooled
        mean/std per feature across all (step, asset) pairs.

        Inputs: P, H, L, V each (T, N), aligned. T must satisfy T > window.
        stride: step between consecutive windows (1 = every step; larger for
        speed at the cost of statistical resolution).
        """
        P = np.asarray(P, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        L = np.asarray(L, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        T, N = P.shape
        if T <= window:
            raise ValueError(f"Need T > window; got T={T}, window={window}.")

        accum = [[] for _ in range(N_FEATURES)]
        for t in range(window, T, stride):
            c_win = P[t - window:t]
            h_win = H[t - window:t]
            l_win = L[t - window:t]
            v_win = V[t - window:t]
            raw = raw_feature_fn(c_win, h_win, l_win, v_win)  # (N, 6)
            for k in range(N_FEATURES):
                col = raw[:, k]
                accum[k].extend(col[~np.isnan(col)].tolist())

        mu = np.zeros(N_FEATURES, dtype=np.float64)
        sigma = np.ones(N_FEATURES, dtype=np.float64)
        for k in range(N_FEATURES):
            arr = np.asarray(accum[k], dtype=np.float64)
            if arr.size < 2:
                continue
            mu[k] = float(np.mean(arr))
            s = float(np.std(arr))
            sigma[k] = s if s > 1e-12 else 1.0

        self.mu = mu
        self.sigma = sigma
        return self

    def transform(self, raw_features):
        """raw_features: (N, 6) raw indicator values. Returns (N, 6) z-scored.
        NaN-safe: NaN raw values produce 0 in the output (matches the policy's
        expected mean-zero unit-variance prior; signals 'no information')."""
        if not self.is_fit:
            raise RuntimeError("FeatureNormalizer not fit. Call fit_from_price_tensor first.")
        raw = np.asarray(raw_features, dtype=np.float64)
        z = (raw - self.mu[None, :]) / self.sigma[None, :]
        z = np.where(np.isnan(z), 0.0, z)
        return z

    def to_dict(self):
        return {
            "feature_names": list(FEATURE_NAMES),
            "mu": self.mu.tolist() if self.mu is not None else None,
            "sigma": self.sigma.tolist() if self.sigma is not None else None,
        }

    @classmethod
    def from_dict(cls, d):
        if list(d.get("feature_names", [])) != list(FEATURE_NAMES):
            raise ValueError(
                f"Feature name mismatch in normalizer stats: "
                f"got {d.get('feature_names')}, expected {list(FEATURE_NAMES)}"
            )
        return cls(mu=d["mu"], sigma=d["sigma"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


# ------------------------------ feature_fn factory -----------------------------
# Returns a closure compatible with PortfolioEnv's feature_fn contract:
# (close_win, high_win, low_win, vol_win) -> flat (6*N,) feature vector.
# The normalizer is captured in the closure so the env signature stays clean.

def make_feature_fn(normalizer):
    """Wrap a fitted FeatureNormalizer into the env's expected feature_fn."""
    if not isinstance(normalizer, FeatureNormalizer):
        raise TypeError("normalizer must be a FeatureNormalizer instance.")
    if not normalizer.is_fit:
        raise RuntimeError("Cannot build feature_fn from an unfit normalizer.")

    def feature_fn(close_win, high_win, low_win, vol_win):
        raw = raw_feature_fn(close_win, high_win, low_win, vol_win)  # (N, 6)
        z = normalizer.transform(raw)                                # (N, 6)
        return z.reshape(-1)                                         # (6*N,)

    return feature_fn


# ------------------------------ legacy default_feature_fn ----------------------
# Kept ONLY so any test fixture or smoke script that imported the old symbol
# still works. New callers should use make_feature_fn(normalizer). This shim
# raises a clear error rather than silently using rolling-window normalization.

def default_feature_fn(*args, **kwargs):
    raise RuntimeError(
        "default_feature_fn has been replaced by make_feature_fn(normalizer). "
        "Fit a FeatureNormalizer on the train price tensor, then call "
        "feature_fn = make_feature_fn(normalizer) and pass it to PortfolioEnv."
    )