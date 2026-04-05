# scripts/train.py
import os
import sys
import json
import time
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


def build_reward(args):
    reward_kwargs = dict(
        W=args.W,
        lam=args.lam,
        eta=args.eta,
        alpha=args.alpha,
        dd_limit=args.dd_limit,
        kappa=args.kappa,
        bonus=args.bonus,
    )
    return make_reward(args.reward_name, **reward_kwargs), reward_kwargs


def build_env(df: pd.DataFrame, args, reward_fn, random_start: bool):
    env = PortfolioEnv(
        prices=df,
        window=args.window,
        trans_cost=args.trans_cost,
        cost_model=args.cost_model,
        episode_length=args.episode_length,
        random_start=random_start,
        reward_fn=reward_fn,
        seed=args.seed,
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


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_csv", default="data/price_train_data.csv")
    p.add_argument("--val_csv", default="data/price_val_data.csv")

    # env
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--trans_cost", type=float, default=0.0)
    p.add_argument("--cost_model", type=str, default="trade_l1", choices=["trade_0", "trade_l1", "trade_l2"])
    p.add_argument("--episode_length", type=int, default=None)
    p.add_argument("--random_start", action="store_true")

    # reward selection + params (used depending on reward)
    p.add_argument("--reward_name", type=str, default="baseline")
    p.add_argument("--W", type=int, default=60)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--dd_limit", type=float, default=-0.10)
    p.add_argument("--kappa", type=float, default=5.0)
    p.add_argument("--bonus", type=float, default=1.0)

    # RL
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)

    # output
    p.add_argument("--out_dir", type=str, default=None)

    args = p.parse_args()

    # output dir
    if args.out_dir is None:
        run_name = f"{args.reward_name}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
        args.out_dir = os.path.join("runs", run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # load data
    train_df = load_prices(args.train_csv)
    val_df = load_prices(args.val_csv)

    # reward
    reward_fn, reward_kwargs = build_reward(args)

    # envs
    train_env = build_env(train_df, args, reward_fn, random_start=args.random_start)
    val_env = build_env(val_df, args, reward_fn, random_start=False)

    # model
    model = PPO("MlpPolicy", train_env, seed=args.seed, verbose=1)

    # train
    model.learn(total_timesteps=args.total_timesteps)

    # save model + config
    model_path = os.path.join(args.out_dir, "model.zip")
    model.save(model_path)

    cfg = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "env": {
            "window": args.window,
            "trans_cost": args.trans_cost,
            "cost_model": args.cost_model,
            "episode_length": args.episode_length,
            "random_start": args.random_start,
        },
        "reward": {
            "name": args.reward_name,
            "kwargs": reward_kwargs,
        },
        "rl": {
            "algo": "PPO",
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # quick val rollout + metrics
    val_hist = rollout(model, val_env, deterministic=True)
    val_hist.to_csv(os.path.join(args.out_dir, "val_history.csv"), index=False)

    equity = pd.Series(val_hist["value"].to_numpy(), name="rl_portfolio")
    mt = metric_table(equity.to_frame())
    tc = turnover_cost_table(val_hist)
    out = pd.concat([mt, tc.T], axis=1)  # quick readable dump
    out.to_csv(os.path.join(args.out_dir, "val_metrics.csv"))

    # print key numbers
    print("\n[VAL]")
    print(mt.round(6).to_string())
    print("\n[VAL turnover/cost]")
    print(tc.round(6).to_string(index=False))
    print(f"\nSaved run to: {args.out_dir}")


if __name__ == "__main__":
    main()
