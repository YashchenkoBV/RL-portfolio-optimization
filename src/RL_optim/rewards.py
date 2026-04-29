# rewards.py
import numpy as np

from metrics import (
    compute_window_ulcer,
    compute_window_avg_dd_duration,
    compute_window_mdd_demeaned,
    compute_window_es,
)

__all__ = [
    "make_reward",
    "reward_log_growth_turnover",
    "reward_log_growth_minus_cii",
    "reward_log_growth_plus_delta_csi",
    "reward_log_growth_minus_delta_ulcer",
    "reward_log_growth_minus_delta_avg_dur",
    "reward_log_growth_minus_delta_mdd_demeaned",
    "reward_log_growth_minus_delta_es",
    "reward_dsr",
    "reward_dsr_plus_delta_csi",
    "reward_with_dd_guardrail",
    "reward_terminal_csi_bonus",
]


# ---------- small internals ----------

def _get_values(ctx, include_current=True, window=None):
    h = (ctx.get("history") or {})
    vals = list(h.get("excess_value", []))

    if include_current:
        vals.append(float(ctx["excess_value"]))
    if window is not None and len(vals) > window:
        vals = vals[-int(window):]

    return np.asarray(vals, dtype=np.float64)


def _csi_1d(values):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = v.size
    if n < 2:
        return 100.0  # no penalty early
    M = (v[:, None] >= v[None, :])
    L = np.tri(n, k=0, dtype=bool)
    s = (M & L).sum(axis=1) - 1
    num = float(s.sum())
    den = float(n * (n - 1) / 2.0)
    return 100.0 * (num / den)


def _csi_from_ctx(ctx, window=None, include_current=True):
    vals = _get_values(ctx, include_current=include_current, window=window)
    return _csi_1d(vals)


def _cii_from_ctx(ctx, window=None, include_current=True):
    return 100.0 - _csi_from_ctx(ctx, window=window, include_current=include_current)


def _delta_metric_from_ctx(ctx, metric_fn, window=60):
    """Compute metric_fn on [history+current] window minus metric_fn on history-only window.

    Returns None if raw history length < window (warmup signal — caller falls back).
    metric_fn signature: (1d array-like) -> float.
    """
    hist_len = len((ctx.get("history") or {}).get("excess_value", []))
    if hist_len < window:
        return None
    vals_with    = _get_values(ctx, include_current=True,  window=window)
    vals_without = _get_values(ctx, include_current=False, window=window)
    return float(metric_fn(vals_with)) - float(metric_fn(vals_without))


# ---------- rewards (ctx -> float) ----------

def reward_log_growth_turnover(ctx, eta=0.0):
    return float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])


def reward_log_growth_minus_cii(ctx, W=60, lam=1.0, eta=0.0):
    # log_growth - lam * (CII/100) - eta * turnover
    cii = _cii_from_ctx(ctx, window=W, include_current=True)
    return float(ctx["log_growth"]) - float(lam) * (cii / 100.0) - float(eta) * float(ctx["turnover"])


def reward_log_growth_plus_delta_csi(ctx, W=60, alpha=1.0, eta=0.0):
    # log_growth + alpha * (CSI_t - CSI_{t-1})/100 - eta * turnover
    # Gate on raw history length: when hist_len == W-1, include_current=True yields W values
    # but include_current=False yields W-1 — denominators differ and delta is mis-scaled at that
    # boundary. Requiring hist_len >= W ensures both windows are exactly length W.
    hist_len = len((ctx.get("history") or {}).get("excess_value", []))
    if hist_len < W:
        return float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])
    csi_t = _csi_from_ctx(ctx, window=W, include_current=True)
    csi_prev = _csi_from_ctx(ctx, window=W, include_current=False)
    delta = (csi_t - csi_prev) / 100.0
    return float(ctx["log_growth"]) + float(alpha) * float(delta) - float(eta) * float(ctx["turnover"])


def reward_with_dd_guardrail(ctx, W=60, dd_limit=-0.10, kappa=5.0, lam=1.0, eta=0.0):
    # penalty if drawdown breaches dd_limit (dd is negative): breach = max(0, dd_limit - dd)
    dd = float(ctx["drawdown"])
    breach = max(0.0, float(dd_limit) - dd)

    cii = _cii_from_ctx(ctx, window=W, include_current=True)
    return (
        float(ctx["log_growth"])
        - float(eta) * float(ctx["turnover"])
        - float(lam) * (cii / 100.0)
        - float(kappa) * breach
    )


def reward_terminal_csi_bonus(ctx, W=None, bonus=1.0, eta=0.0):
    # step reward + optional terminal bonus if ctx["is_terminal"] is provided by caller/env
    r = float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])
    if ctx.get("is_terminal", False):
        csi = _csi_from_ctx(ctx, window=W, include_current=True)
        r += float(bonus) * (csi / 100.0)
    return r


# ---------- Phase 1 Δ-metric rewards ----------
# Sign convention: window metrics (ulcer, avg_dur, mdd_demeaned, es) are all
# "higher = worse risk", so we subtract β * Δmetric. No /100 for fraction-valued
# metrics — that scaling is absorbed by the calibration phase.

def reward_log_growth_minus_delta_ulcer(ctx, W=60, beta=1.0, eta=0.0):
    base = float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])
    delta = _delta_metric_from_ctx(ctx, compute_window_ulcer, window=W)
    if delta is None:
        return base
    return base - float(beta) * float(delta)


def reward_log_growth_minus_delta_avg_dur(ctx, W=60, beta=1.0, eta=0.0):
    base = float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])
    delta = _delta_metric_from_ctx(ctx, compute_window_avg_dd_duration, window=W)
    if delta is None:
        return base
    return base - float(beta) * float(delta)


def reward_log_growth_minus_delta_mdd_demeaned(ctx, W=60, beta=1.0, eta=0.0):
    base = float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])
    delta = _delta_metric_from_ctx(ctx, compute_window_mdd_demeaned, window=W)
    if delta is None:
        return base
    return base - float(beta) * float(delta)


def reward_log_growth_minus_delta_es(ctx, W=60, beta=1.0, eta=0.0, alpha_tail=0.05):
    # alpha_tail is the ES tail probability (distinct from the `alpha` ΔCSI coefficient)
    base = float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])
    metric_fn = lambda s: compute_window_es(s, alpha=alpha_tail)
    delta = _delta_metric_from_ctx(ctx, metric_fn, window=W)
    if delta is None:
        return base
    return base - float(beta) * float(delta)


def reward_dsr(ctx, eta=0.0):
    # Differential Sharpe Ratio (Moody & Saffell). Uses pre-update EMA state from env.
    # var floor is mandatory: at episode start A=B=0 collapses var; floor keeps
    # var**1.5 numerically bounded for the first ~10 steps.
    A = float(ctx["dsr_A"])
    B = float(ctx["dsr_B"])
    R = float(ctx["portfolio_return"])
    delta_A = R - A
    delta_B = R * R - B
    var = B - A * A
    var = max(var, 1e-8)
    dsr = (B * delta_A - 0.5 * A * delta_B) / (var ** 1.5)
    return dsr - float(eta) * float(ctx["turnover"])


def reward_dsr_plus_delta_csi(ctx, W=60, alpha=1.0, eta=0.0):
    # base already includes -eta*turnover; do not subtract it twice
    base = reward_dsr(ctx, eta=eta)
    hist_len = len((ctx.get("history") or {}).get("excess_value", []))
    if hist_len < W:
        return base
    csi_t    = _csi_from_ctx(ctx, window=W, include_current=True)
    csi_prev = _csi_from_ctx(ctx, window=W, include_current=False)
    delta = (csi_t - csi_prev) / 100.0
    return base + float(alpha) * float(delta)


def _pick_kwargs(kwargs, allowed):
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_reward(name, **kwargs):
    name = str(name).lower()

    if name in ("baseline", "log_growth", "log_growth_turnover"):
        kw = _pick_kwargs(kwargs, {"eta"})
        return lambda ctx: reward_log_growth_turnover(ctx, **kw)

    if name in ("csi_level", "minus_cii", "log_growth_minus_cii"):
        kw = _pick_kwargs(kwargs, {"W", "lam", "eta"})
        return lambda ctx: reward_log_growth_minus_cii(ctx, **kw)

    if name in ("csi_delta", "delta_csi", "log_growth_plus_delta_csi"):
        kw = _pick_kwargs(kwargs, {"W", "alpha", "eta"})
        return lambda ctx: reward_log_growth_plus_delta_csi(ctx, **kw)

    if name in ("ulcer_delta", "delta_ulcer"):
        kw = _pick_kwargs(kwargs, {"W", "beta", "eta"})
        return lambda ctx: reward_log_growth_minus_delta_ulcer(ctx, **kw)

    if name in ("avg_dur_delta", "delta_avg_dur"):
        kw = _pick_kwargs(kwargs, {"W", "beta", "eta"})
        return lambda ctx: reward_log_growth_minus_delta_avg_dur(ctx, **kw)

    if name in ("mdd_demeaned_delta", "delta_mdd_demeaned"):
        kw = _pick_kwargs(kwargs, {"W", "beta", "eta"})
        return lambda ctx: reward_log_growth_minus_delta_mdd_demeaned(ctx, **kw)

    if name in ("es_delta", "delta_es"):
        kw = _pick_kwargs(kwargs, {"W", "beta", "eta", "alpha_tail"})
        return lambda ctx: reward_log_growth_minus_delta_es(ctx, **kw)

    if name in ("dsr",):
        kw = _pick_kwargs(kwargs, {"eta"})
        return lambda ctx: reward_dsr(ctx, **kw)

    if name in ("dsr_csi_delta", "dsr_plus_delta_csi"):
        kw = _pick_kwargs(kwargs, {"W", "alpha", "eta"})
        return lambda ctx: reward_dsr_plus_delta_csi(ctx, **kw)

    if name in ("dd_guardrail", "guardrail", "dd_csi"):
        kw = _pick_kwargs(kwargs, {"W", "dd_limit", "kappa", "lam", "eta"})
        return lambda ctx: reward_with_dd_guardrail(ctx, **kw)

    if name in ("terminal_csi", "terminal_bonus"):
        kw = _pick_kwargs(kwargs, {"W", "bonus", "eta"})
        return lambda ctx: reward_terminal_csi_bonus(ctx, **kw)

    raise ValueError(f"Unknown reward name: {name}")
