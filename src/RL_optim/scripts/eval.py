# scripts/eval.py
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
from rewards import make_reward
from metrics import metric_table, turnover_cost_table
from indicators import FeatureNormalizer, make_feature_fn
from rollouts import threaded_rollout

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def load_ohlcv(prefix: str) -> dict:
    """prefix is e.g. 'data/price_test' — loads four CSVs."""
    return {
        "close":  pd.read_csv(f"{prefix}_close.csv"),
        "high":   pd.read_csv(f"{prefix}_high.csv"),
        "low":    pd.read_csv(f"{prefix}_low.csv"),
        "volume": pd.read_csv(f"{prefix}_volume.csv"),
    }


def _resolve_run_dir_and_model(model_path, prefer_best=True):
    """Accept either a path to a .zip model or a run directory.
    Returns (run_dir, resolved_model_path)."""
    if os.path.isdir(model_path):
        run_dir = model_path
        best = os.path.join(run_dir, "best_model.zip")
        final = os.path.join(run_dir, "model.zip")
        if prefer_best and os.path.exists(best):
            return run_dir, best
        if os.path.exists(final):
            return run_dir, final
        raise FileNotFoundError(f"No model.zip or best_model.zip in {run_dir}")
    # explicit file path: take its dirname as run_dir
    return os.path.dirname(model_path), model_path


def load_config(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return None


def build_reward_from_cfg(cfg_reward):
    return make_reward(cfg_reward["name"], **cfg_reward.get("kwargs", {}))


def build_env(ohlcv: dict, env_cfg, reward_fn, feature_fn, seed=0):
    env = PortfolioEnv(
        prices=ohlcv,
        window=env_cfg["window"],
        trans_cost=env_cfg["trans_cost"],
        cost_model=env_cfg["cost_model"],
        episode_length=None,
        random_start=False,
        reward_fn=reward_fn,
        feature_fn=feature_fn,
        seed=seed,
        initial_value=1.0,
        rf_rate=env_cfg.get("rf_rate", 0.0)
    )
    return Monitor(env)


def _model_class_from_cfg(cfg):
    algo = cfg.get("rl", {}).get("algo", "PPO")
    if algo == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO
    return PPO


def equal_weight_baseline(df: pd.DataFrame, window: int, trans_cost: float, cost_model: str, rf_rate: float = 0.0):
    # match env time convention: decide at t, apply P[t+1]/P[t]-1
    if "Date" in df.columns:
        dates = df["Date"].to_numpy()
        P = df.drop(columns=["Date"]).to_numpy(dtype=np.float64)
    else:
        dates = np.arange(len(df))
        P = df.to_numpy(dtype=np.float64)

    T, N = P.shape
    start = window
    end = T - 1  # inclusive price index (last step uses P[end]/P[end-1])

    w_target = np.ones(N, dtype=np.float64) / N
    w_prev = w_target.copy()

    cash_value = 1.0
    rf_daily = rf_rate / 252

    value = 1.0
    hist = {"date": [], "value": [], "excess_value": [], "port_ret": [], "turnover": [], "cost": [], "dd": []}
    peak = 1.0

    for t in range(start, end):
        r = P[t + 1] / P[t] - 1.0
        port_ret = float(np.dot(w_target, r))

        turnover = float(np.sum(np.abs(w_target - w_prev)))
        if cost_model == "trade_0" or trans_cost <= 0:
            cost = 0.0
        elif cost_model == "trade_l2":
            cost = trans_cost * (turnover ** 2)
        else:
            cost = trans_cost * turnover

        value = value * max(1e-12, (1.0 + port_ret)) * max(0.0, (1.0 - cost))
        if value > peak:
            peak = value
        dd = float(value / peak - 1.0)

        cash_value *= (1 + rf_daily)
        excess_value = value / cash_value

        hist["date"].append(dates[t + 1] if (t + 1) < len(dates) else dates[t])
        hist["value"].append(value)
        hist["excess_value"].append(excess_value)
        hist["port_ret"].append(port_ret)
        hist["turnover"].append(turnover)
        hist["cost"].append(cost)
        hist["dd"].append(dd)

        w_after = w_target * (1.0 + r)
        w_sum = w_after.sum()
        w_prev = w_after / w_sum if w_sum > 1e-12 else w_prev

    return pd.DataFrame(hist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to a run directory or a specific .zip model file.")
    p.add_argument("--test_prefix", default="data/price_test")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefer_best", type=int, default=1, choices=[0, 1],
                   help="If --model_path is a run dir, prefer best_model.zip over model.zip.")
    args = p.parse_args()

    run_dir, model_path = _resolve_run_dir_and_model(args.model_path, prefer_best=bool(args.prefer_best))

    cfg = load_config(run_dir)
    if cfg is None:
        raise FileNotFoundError(f"config.json not found at {run_dir}.")

    if args.out_dir is None:
        args.out_dir = os.path.join(run_dir, "eval")
    os.makedirs(args.out_dir, exist_ok=True)

    test_ohlcv = load_ohlcv(args.test_prefix)

    # load feature normalizer fit at train time. Must exist next to the model;
    # raising here instead of silently fitting on test prevents look-ahead bias.
    normalizer_path = os.path.join(run_dir, "feature_normalizer.json")
    if not os.path.exists(normalizer_path):
        raise FileNotFoundError(
            f"feature_normalizer.json not found at {normalizer_path}. "
            "This file is produced by train.py and is required for eval."
        )
    normalizer = FeatureNormalizer.load(normalizer_path)
    feature_fn = make_feature_fn(normalizer)

    reward_fn = build_reward_from_cfg(cfg["reward"])
    env_cfg = cfg["env"]

    env = build_env(test_ohlcv, env_cfg, reward_fn, feature_fn, seed=args.seed)

    ModelCls = _model_class_from_cfg(cfg)
    model = ModelCls.load(model_path, env=env)
    print(f"[eval] loaded {ModelCls.__name__} from {model_path}")

    hist = threaded_rollout(model, env, deterministic=True)
    hist.to_csv(os.path.join(args.out_dir, "eval_history.csv"), index=False)

    rl_equity = pd.Series(hist["value"].to_numpy(), name="rl_portfolio")
    rl_excess = pd.Series(hist["excess_value"].to_numpy(), name="rl_portfolio") if "excess_value" in hist.columns else None
    rl_metrics = metric_table(rl_equity.to_frame(), excess_prices=rl_excess)
    rl_costs = turnover_cost_table(hist)

    # baseline: equal-weight uses close only
    b_hist = equal_weight_baseline(
        test_ohlcv["close"],
        window=env_cfg["window"],
        trans_cost=env_cfg["trans_cost"],
        cost_model=env_cfg["cost_model"],
        rf_rate=env_cfg.get("rf_rate", 0.0)
    )
    b_hist.to_csv(os.path.join(args.out_dir, "baseline_equal_weight_history.csv"), index=False)

    b_equity = pd.Series(b_hist["value"].to_numpy(), name="equal_weight")
    b_excess = pd.Series(b_hist["excess_value"].to_numpy(), name="equal_weight") if "excess_value" in b_hist.columns else None
    b_metrics = metric_table(b_equity.to_frame(), excess_prices=b_excess)
    b_costs = turnover_cost_table(b_hist)

    rl_metrics.to_csv(os.path.join(args.out_dir, "eval_metrics.csv"))
    rl_costs.to_csv(os.path.join(args.out_dir, "eval_turnover_costs.csv"), index=False)
    b_metrics.to_csv(os.path.join(args.out_dir, "baseline_equal_weight_metrics.csv"))
    b_costs.to_csv(os.path.join(args.out_dir, "baseline_equal_weight_turnover_costs.csv"), index=False)

    print("\n[TEST RL]")
    print(rl_metrics.round(6).to_string())
    print("\n[TEST RL turnover/cost]")
    print(rl_costs.round(6).to_string(index=False))

    print("\n[TEST Equal-weight baseline]")
    print(b_metrics.round(6).to_string())
    print("\n[TEST Equal-weight turnover/cost]")
    print(b_costs.round(6).to_string(index=False))

    print(f"\nSaved eval to: {args.out_dir}")


if __name__ == "__main__":
    main()
