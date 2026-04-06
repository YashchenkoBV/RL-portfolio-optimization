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

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def load_prices(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_config(model_path: str):
    run_dir = os.path.dirname(model_path)
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return None


def build_reward_from_cfg(cfg_reward):
    return make_reward(cfg_reward["name"], **cfg_reward.get("kwargs", {}))


def build_env(df: pd.DataFrame, env_cfg, reward_fn, seed=0):
    env = PortfolioEnv(
        prices=df,
        window=env_cfg["window"],
        trans_cost=env_cfg["trans_cost"],
        cost_model=env_cfg["cost_model"],
        episode_length=None,
        random_start=False,
        reward_fn=reward_fn,
        seed=seed,
        initial_value=1.0,
    )
    return Monitor(env)


def rollout(model, env, deterministic=True):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.unwrapped.get_history_df()


def equal_weight_baseline(df: pd.DataFrame, window: int, trans_cost: float, cost_model: str):
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

    value = 1.0
    hist = {"date": [], "value": [], "port_ret": [], "turnover": [], "cost": [], "dd": []}
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

        hist["date"].append(dates[t + 1] if (t + 1) < len(dates) else dates[t])
        hist["value"].append(value)
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
    p.add_argument("--model_path", required=True)
    p.add_argument("--test_csv", default="data/price_test_data.csv")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.model_path)
    if cfg is None:
        raise FileNotFoundError("config.json not found next to model. Provide a run directory with config.json.")

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.model_path), "eval")
    os.makedirs(args.out_dir, exist_ok=True)

    test_df = load_prices(args.test_csv)

    reward_fn = build_reward_from_cfg(cfg["reward"])
    env_cfg = cfg["env"]

    env = build_env(test_df, env_cfg, reward_fn, seed=args.seed)

    model = PPO.load(args.model_path, env=env)

    hist = rollout(model, env, deterministic=True)
    hist.to_csv(os.path.join(args.out_dir, "eval_history.csv"), index=False)

    rl_equity = pd.Series(hist["value"].to_numpy(), name="rl_portfolio")
    rl_metrics = metric_table(rl_equity.to_frame())
    rl_costs = turnover_cost_table(hist)

    # baseline: equal-weight (same cost model)
    b_hist = equal_weight_baseline(
        test_df,
        window=env_cfg["window"],
        trans_cost=env_cfg["trans_cost"],
        cost_model=env_cfg["cost_model"],
    )
    b_hist.to_csv(os.path.join(args.out_dir, "baseline_equal_weight_history.csv"), index=False)

    b_equity = pd.Series(b_hist["value"].to_numpy(), name="equal_weight")
    b_metrics = metric_table(b_equity.to_frame())
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
