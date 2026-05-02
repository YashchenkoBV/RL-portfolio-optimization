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
from indicators import FeatureNormalizer, make_feature_fn
from rollouts import threaded_rollout
from callbacks import ValidationCallback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def load_ohlcv(prefix: str) -> dict:
    """prefix is e.g. 'data/price_train' — loads four CSVs."""
    return {
        "close":  pd.read_csv(f"{prefix}_close.csv"),
        "high":   pd.read_csv(f"{prefix}_high.csv"),
        "low":    pd.read_csv(f"{prefix}_low.csv"),
        "volume": pd.read_csv(f"{prefix}_volume.csv"),
    }


def build_reward(args):
    reward_kwargs = dict(
        W=args.W,
        lam=args.lam,
        eta=args.eta,
        alpha=args.alpha,
        beta=args.beta,
        dd_limit=args.dd_limit,
        kappa=args.kappa,
        bonus=args.bonus,
    )
    return make_reward(args.reward_name, **reward_kwargs), reward_kwargs


def build_env(ohlcv, args, reward_fn, feature_fn, *, random_start, episode_length):
    env = PortfolioEnv(
        prices=ohlcv,
        window=args.window,
        trans_cost=args.trans_cost,
        cost_model=args.cost_model,
        episode_length=episode_length,
        random_start=random_start,
        reward_fn=reward_fn,
        feature_fn=feature_fn,
        seed=args.seed,
        initial_value=1.0,
        rf_rate=args.rf_rate,
    )
    return Monitor(env)


def _apply_calibration(args):
    """Load calibration.json (if present) and override args.alpha / args.beta
    when the user did NOT pass them explicitly (we detect this by sentinel
    default of None). Returns (calibration_loaded, field, value) where field
    is "alpha" / "beta" / None.
    """
    cal_loaded = False
    used_field = None
    used_value = None

    if args.calibration_path and os.path.exists(args.calibration_path):
        with open(args.calibration_path, "r") as f:
            cal = json.load(f)
        cal_loaded = True
        coef_table = cal.get("coef_table", {})
        row = coef_table.get(args.reward_name)
        if row is not None:
            if "alpha" in row and args.alpha is None:
                args.alpha = float(row["alpha"])
                used_field, used_value = "alpha", args.alpha
                print(f"[calibration] {args.reward_name}: alpha = {used_value:.6g} (source: {row.get('source')})")
            elif "beta" in row and args.beta is None:
                args.beta = float(row["beta"])
                used_field, used_value = "beta", args.beta
                print(f"[calibration] {args.reward_name}: beta = {used_value:.6g} (source: {row.get('source')})")
        else:
            print(f"[calibration] '{args.reward_name}' not in coef_table — no override applied")
    else:
        print(f"[calibration] no calibration file at {args.calibration_path} — using defaults")

    # Backstop fallbacks for any reward that needs alpha/beta but did not get
    # an override (e.g. user passed a name with no calibration entry).
    if args.alpha is None:
        args.alpha = 1.0
    if args.beta is None:
        args.beta = 1.0

    return cal_loaded, used_field, used_value


def _to_array(df):
    return df.drop(columns=["Date"]).to_numpy(dtype=np.float64) if "Date" in df.columns \
        else df.to_numpy(dtype=np.float64)


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_prefix", default="data/price_train")
    p.add_argument("--val_prefix", default="data/price_val")

    # env
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--trans_cost", type=float, default=0.001)
    p.add_argument("--cost_model", type=str, default="trade_l1",
                   choices=["trade_0", "trade_l1", "trade_l2"])
    p.add_argument("--episode_length", type=int, default=252)
    p.add_argument("--no_random_start", dest="random_start", action="store_false")
    p.set_defaults(random_start=True)
    p.add_argument("--rf_rate", type=float, default=0.04)

    # reward selection + params (used depending on reward)
    p.add_argument("--reward_name", type=str, default="baseline")
    p.add_argument("--W", type=int, default=60)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    # alpha / beta default to None so calibration loading can detect "not passed".
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--dd_limit", type=float, default=-0.10)
    p.add_argument("--kappa", type=float, default=5.0)
    p.add_argument("--bonus", type=float, default=1.0)

    # calibration
    p.add_argument("--calibration_path", type=str, default="data/calibration.json")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Recorded but not applied; coefs in calibration.json are pre-computed for γ=1.")

    # RL
    p.add_argument("--policy", type=str, default="mlp", choices=["mlp", "lstm"])
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_freq", type=int, default=50_000)

    # LSTM hyperparams (only used when --policy lstm)
    p.add_argument("--lstm_hidden_size", type=int, default=128)
    p.add_argument("--n_lstm_layers", type=int, default=1)
    p.add_argument("--enable_critic_lstm", type=int, default=1, choices=[0, 1])
    p.add_argument("--shared_lstm", type=int, default=0, choices=[0, 1])

    # output
    p.add_argument("--out_dir", type=str, default=None)

    args = p.parse_args()

    # Calibration override (must run before build_reward — alpha/beta feed in there).
    cal_loaded, cal_field, cal_value = _apply_calibration(args)

    # output dir
    if args.out_dir is None:
        run_name = f"{args.reward_name}_{args.policy}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
        args.out_dir = os.path.join("runs", run_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # load data
    train_ohlcv = load_ohlcv(args.train_prefix)
    val_ohlcv = load_ohlcv(args.val_prefix)

    # fit feature normalizer on TRAIN ONLY; reuse same stats at val/test time.
    normalizer = FeatureNormalizer().fit_from_price_tensor(
        P=_to_array(train_ohlcv["close"]),
        H=_to_array(train_ohlcv["high"]),
        L=_to_array(train_ohlcv["low"]),
        V=_to_array(train_ohlcv["volume"]),
        window=args.window,
    )
    normalizer.save(os.path.join(args.out_dir, "feature_normalizer.json"))
    feature_fn = make_feature_fn(normalizer)

    # reward
    reward_fn, reward_kwargs = build_reward(args)

    # envs — train uses episode_length + random_start; val is full deterministic rollout
    train_env = build_env(
        train_ohlcv, args, reward_fn, feature_fn,
        random_start=args.random_start, episode_length=args.episode_length,
    )
    val_env = build_env(
        val_ohlcv, args, reward_fn, feature_fn,
        random_start=False, episode_length=None,
    )

    # model
    common_hyper = dict(
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.995,
        ent_coef=0.01,
        clip_range=0.1,
    )

    if args.policy == "lstm":
        from sb3_contrib import RecurrentPPO
        policy_kwargs = dict(
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=args.n_lstm_layers,
            enable_critic_lstm=bool(args.enable_critic_lstm),
            shared_lstm=bool(args.shared_lstm),
        )
        model = RecurrentPPO(
            "MlpLstmPolicy", train_env, seed=args.seed, verbose=1,
            policy_kwargs=policy_kwargs, **common_hyper,
        )
        algo_name = "RecurrentPPO"
    else:
        model = PPO(
            "MlpPolicy", train_env, seed=args.seed, verbose=1, **common_hyper,
        )
        algo_name = "PPO"

    # validation callback
    val_cb = ValidationCallback(val_env=val_env, out_dir=args.out_dir, eval_freq=args.eval_freq)

    # train
    model.learn(total_timesteps=args.total_timesteps, callback=val_cb)

    # save final model + config
    model_path = os.path.join(args.out_dir, "model.zip")
    model.save(model_path)

    cfg = {
        "train_prefix": args.train_prefix,
        "val_prefix": args.val_prefix,
        "env": {
            "window": args.window,
            "trans_cost": args.trans_cost,
            "cost_model": args.cost_model,
            "episode_length": args.episode_length,
            "random_start": args.random_start,
            "rf_rate": args.rf_rate,
        },
        "reward": {
            "name": args.reward_name,
            "kwargs": reward_kwargs,
            "calibration": {
                "loaded": cal_loaded,
                "path": args.calibration_path,
                "gamma": args.gamma,
                "field": cal_field,
                "value": cal_value,
            },
        },
        "rl": {
            "algo": algo_name,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "eval_freq": args.eval_freq,
            "policy": args.policy,
            "hyper": dict(common_hyper),
            "lstm": {
                "lstm_hidden_size": args.lstm_hidden_size,
                "n_lstm_layers": args.n_lstm_layers,
                "enable_critic_lstm": bool(args.enable_critic_lstm),
                "shared_lstm": bool(args.shared_lstm),
            } if args.policy == "lstm" else None,
        },
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # final val rollout — produces the equity-curve artifact (the callback only
    # writes per-eval metric rows, not the per-step history).
    val_hist = threaded_rollout(model, val_env, deterministic=True)
    val_hist.to_csv(os.path.join(args.out_dir, "val_history.csv"), index=False)

    equity = pd.Series(val_hist["value"].to_numpy(), name="rl_portfolio")
    mt = metric_table(equity.to_frame())
    tc = turnover_cost_table(val_hist)
    out = pd.concat([mt, tc.T], axis=1)
    out.to_csv(os.path.join(args.out_dir, "val_metrics.csv"))

    print("\n[VAL]")
    print(mt.round(6).to_string())
    print("\n[VAL turnover/cost]")
    print(tc.round(6).to_string(index=False))
    print(f"\nSaved run to: {args.out_dir}")


if __name__ == "__main__":
    main()
