# scripts/calibrate.py
"""σ-matching calibration sweep. Outputs data/calibration.json.

Two baselines (equal-weight, random-init MLP PPO) sweep the train split
through a recording reward that captures every Δ-metric component at each
step. Per-metric max σ is taken across baselines, then per-variant
coefficients are derived as γ · σ_base / σ_metric. γ is locked at 1.0.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from portfolio_env import PortfolioEnv
from indicators import FeatureNormalizer, make_feature_fn
from rewards import _csi_from_ctx, _delta_metric_from_ctx, reward_dsr
from metrics import (
    compute_window_ulcer, compute_window_avg_dd_duration,
    compute_window_mdd_demeaned, compute_window_es,
)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def load_ohlcv(prefix: str) -> dict:
    return {
        "close":  pd.read_csv(f"{prefix}_close.csv"),
        "high":   pd.read_csv(f"{prefix}_high.csv"),
        "low":    pd.read_csv(f"{prefix}_low.csv"),
        "volume": pd.read_csv(f"{prefix}_volume.csv"),
    }


def _to_array(df):
    return df.drop(columns=["Date"]).to_numpy(dtype=np.float64) if "Date" in df.columns \
        else df.to_numpy(dtype=np.float64)


def make_recording_reward(records, W=60):
    def reward_fn(ctx):
        rec = {
            "log_growth": float(ctx["log_growth"]),
            "dsr":        float(reward_dsr(ctx, eta=0.0)),
        }
        hist_len = len((ctx.get("history") or {}).get("excess_value", []))
        if hist_len >= W:
            csi_t = _csi_from_ctx(ctx, window=W, include_current=True)
            csi_p = _csi_from_ctx(ctx, window=W, include_current=False)
            rec["delta_csi"] = (csi_t - csi_p) / 100.0
        else:
            rec["delta_csi"] = None
        rec["delta_ulcer"]         = _delta_metric_from_ctx(ctx, compute_window_ulcer, window=W)
        rec["delta_avg_dur"]       = _delta_metric_from_ctx(ctx, compute_window_avg_dd_duration, window=W)
        rec["delta_mdd_demeaned"]  = _delta_metric_from_ctx(ctx, compute_window_mdd_demeaned, window=W)
        rec["delta_es"]            = _delta_metric_from_ctx(ctx, lambda s: compute_window_es(s, alpha=0.05), window=W)
        records.append(rec)
        return float(ctx["log_growth"])  # value irrelevant for calibration
    return reward_fn


def _build_env(ohlcv, feature_fn, reward_fn, args):
    return PortfolioEnv(
        prices=ohlcv,
        window=args.window,
        trans_cost=args.trans_cost,
        cost_model=args.cost_model,
        episode_length=None,
        random_start=False,
        reward_fn=reward_fn,
        feature_fn=feature_fn,
        seed=args.seed,
        initial_value=1.0,
        rf_rate=args.rf_rate,
    )


def _equal_weight_action(N_plus_1):
    # softmax with cash logit at -100 → ~uniform across N assets, ~0 cash weight
    a = np.zeros(N_plus_1, dtype=np.float32)
    a[-1] = -100.0
    return a


def sweep_equal_weight(ohlcv, feature_fn, args):
    records = []
    env = _build_env(ohlcv, feature_fn, make_recording_reward(records, W=args.W), args)
    obs, _info = env.reset()
    done = False
    a = _equal_weight_action(env.action_space.shape[0])
    while not done:
        obs, _r, terminated, truncated, _info = env.step(a)
        done = terminated or truncated
    return records


def sweep_random_ppo(ohlcv, feature_fn, args):
    records = []
    sweep_env = _build_env(ohlcv, feature_fn, make_recording_reward(records, W=args.W), args)
    probe_env = _build_env(ohlcv, feature_fn, None, args)
    model = PPO("MlpPolicy", Monitor(probe_env), seed=args.seed, verbose=0)
    obs, _info = sweep_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _r, terminated, truncated, _info = sweep_env.step(action)
        done = terminated or truncated
    return records


def _compute_sigmas(records):
    df = pd.DataFrame(records)
    cols = ["log_growth", "dsr", "delta_csi", "delta_ulcer",
            "delta_avg_dur", "delta_mdd_demeaned", "delta_es"]
    sig = {}
    for c in cols:
        if c not in df.columns:
            sig[c] = float("nan")
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        sig[c] = float(s.std(ddof=0)) if len(s) >= 2 else float("nan")
    return sig


def _selected_max(eq, rd):
    return {k: float(max(eq[k], rd[k])) for k in eq.keys()}


def _build_coef_table(selected, gamma):
    sl = selected["log_growth"]
    sd = selected["dsr"]
    return {
        "csi_delta":          {"alpha": float(gamma * sl / selected["delta_csi"]),          "source": "log_growth"},
        "ulcer_delta":        {"beta":  float(gamma * sl / selected["delta_ulcer"]),        "source": "log_growth"},
        "avg_dur_delta":      {"beta":  float(gamma * sl / selected["delta_avg_dur"]),      "source": "log_growth"},
        "mdd_demeaned_delta": {"beta":  float(gamma * sl / selected["delta_mdd_demeaned"]), "source": "log_growth"},
        "es_delta":           {"beta":  float(gamma * sl / selected["delta_es"]),           "source": "log_growth"},
        "dsr_csi_delta":      {"alpha": float(gamma * sd / selected["delta_csi"]),          "source": "dsr"},
    }


def run_calibration(args):
    train_ohlcv = load_ohlcv(args.train_prefix)
    normalizer = FeatureNormalizer().fit_from_price_tensor(
        P=_to_array(train_ohlcv["close"]),
        H=_to_array(train_ohlcv["high"]),
        L=_to_array(train_ohlcv["low"]),
        V=_to_array(train_ohlcv["volume"]),
        window=args.window,
    )
    feature_fn = make_feature_fn(normalizer)

    eq_records = sweep_equal_weight(train_ohlcv, feature_fn, args)
    rd_records = sweep_random_ppo(train_ohlcv, feature_fn, args)

    sigma_eq = _compute_sigmas(eq_records)
    sigma_rd = _compute_sigmas(rd_records)

    selected = _selected_max(sigma_eq, sigma_rd)
    coef = _build_coef_table(selected, gamma=1.0)

    payload = {
        "gamma": 1.0,
        "W": args.W,
        "rf_rate": args.rf_rate,
        "trans_cost": args.trans_cost,
        "cost_model": args.cost_model,
        "baselines": {
            "equal_weight": {f"sigma_{k}": v for k, v in sigma_eq.items()},
            "random_ppo":   {f"sigma_{k}": v for k, v in sigma_rd.items()},
        },
        "selected": {f"sigma_{k}": v for k, v in selected.items()},
        "coef_table": coef,
    }
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_prefix", default="data/price_train")
    p.add_argument("--out", default="data/calibration.json")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--W", type=int, default=60)
    p.add_argument("--rf_rate", type=float, default=0.04)
    p.add_argument("--trans_cost", type=float, default=0.001)
    p.add_argument("--cost_model", type=str, default="trade_l1",
                   choices=["trade_0", "trade_l1", "trade_l2"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    payload = run_calibration(args)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved calibration to: {args.out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
