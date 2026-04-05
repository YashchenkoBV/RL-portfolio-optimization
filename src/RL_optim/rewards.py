# rewards.py
import numpy as np

__all__ = [
    "make_reward",
    "reward_log_growth_turnover",
    "reward_log_growth_minus_cii",
    "reward_log_growth_plus_delta_csi",
    "reward_with_dd_guardrail",
    "reward_terminal_csi_bonus",
]


# ---------- small internals ----------

def _get_values(ctx, include_current=True, window=None):
    h = (ctx.get("history") or {})
    vals = list(h.get("value", []))

    if include_current:
        vals.append(float(ctx["value"]))
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


# ---------- rewards (ctx -> float) ----------

def reward_log_growth_turnover(ctx, eta=0.0):
    return float(ctx["log_growth"]) - float(eta) * float(ctx["turnover"])


def reward_log_growth_minus_cii(ctx, W=60, lam=1.0, eta=0.0):
    # log_growth - lam * (CII/100) - eta * turnover
    cii = _cii_from_ctx(ctx, window=W, include_current=True)
    return float(ctx["log_growth"]) - float(lam) * (cii / 100.0) - float(eta) * float(ctx["turnover"])


def reward_log_growth_plus_delta_csi(ctx, W=60, alpha=1.0, eta=0.0):
    # log_growth + alpha * (CSI_t - CSI_{t-1})/100 - eta * turnover
    # During warmup (fewer than W values), delta is undefined — fall back to log_growth only
    vals = _get_values(ctx, include_current=True, window=W)
    if vals.shape[0] < W:
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

    if name in ("dd_guardrail", "guardrail", "dd_csi"):
        kw = _pick_kwargs(kwargs, {"W", "dd_limit", "kappa", "lam", "eta"})
        return lambda ctx: reward_with_dd_guardrail(ctx, **kw)

    if name in ("terminal_csi", "terminal_bonus"):
        kw = _pick_kwargs(kwargs, {"W", "bonus", "eta"})
        return lambda ctx: reward_terminal_csi_bonus(ctx, **kw)

    raise ValueError(f"Unknown reward name: {name}")
