# scripts/verify_lstm_threading.py
"""Manual verification that RecurrentPPO actually accumulates LSTM state at
predict time. Run once after Day 4 setup, before launching the Day 5 pilot.

Usage:
    python scripts/verify_lstm_threading.py

If the LSTM is correctly threaded, the action sequence with state threading
diverges from the stateless version after the first step (where they
necessarily agree, since both start from a zero hidden state). If they
agree at every step, the policy has silently degenerated to a feedforward
mode — the canonical sb3-contrib failure mode — and the pilot must not run
until this is fixed.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from portfolio_env import PortfolioEnv
from indicators import FeatureNormalizer, make_feature_fn

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor


def _synthetic_ohlcv(N=4, T=180, seed=0):
    rng = np.random.RandomState(seed)
    log_ret = rng.normal(0.0005, 0.012, size=(T - 1, N))
    close = np.empty((T, N))
    close[0] = 100.0
    for t in range(1, T):
        close[t] = close[t - 1] * np.exp(log_ret[t - 1])
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, size=(T, N))))
    low  = close * (1.0 - np.abs(rng.normal(0.0, 0.005, size=(T, N))))
    vol  = np.full((T, N), 1e6) + rng.normal(0, 1e4, size=(T, N))

    cols = [f"A{i}" for i in range(N)]
    dates = pd.date_range("2024-01-01", periods=T).strftime("%Y-%m-%d")
    def to_df(arr):
        df = pd.DataFrame(arr, columns=cols)
        df.insert(0, "Date", dates)
        return df
    return {
        "close":  to_df(close),
        "high":   to_df(high),
        "low":    to_df(low),
        "volume": to_df(vol),
    }


def _build_env(ohlcv, feature_fn, seed=0):
    env = PortfolioEnv(
        prices=ohlcv,
        window=60,
        trans_cost=0.001,
        cost_model="trade_l1",
        episode_length=None,
        random_start=False,
        reward_fn=None,
        feature_fn=feature_fn,
        seed=seed,
        initial_value=1.0,
        rf_rate=0.04,
    )
    return Monitor(env)


def _collect_actions(model, env, threaded):
    obs, _ = env.reset()
    actions = []
    state = None
    episode_starts = np.ones((1,), dtype=bool)
    for step in range(10):
        if threaded:
            action, state = model.predict(obs, state=state, episode_start=episode_starts, deterministic=True)
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, state=None,
                                      episode_start=np.ones((1,), dtype=bool),
                                      deterministic=True)
        actions.append(np.asarray(action).reshape(-1).copy())
        obs, _r, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return actions


def main():
    ohlcv = _synthetic_ohlcv(N=4, T=180, seed=0)

    def _to_arr(df):
        return df.drop(columns=["Date"]).to_numpy(dtype=np.float64)

    norm = FeatureNormalizer().fit_from_price_tensor(
        P=_to_arr(ohlcv["close"]), H=_to_arr(ohlcv["high"]),
        L=_to_arr(ohlcv["low"]),   V=_to_arr(ohlcv["volume"]),
        window=60,
    )
    feature_fn = make_feature_fn(norm)

    env_threaded   = _build_env(ohlcv, feature_fn, seed=0)
    env_stateless  = _build_env(ohlcv, feature_fn, seed=0)

    policy_kwargs = dict(
        lstm_hidden_size=128, n_lstm_layers=1,
        enable_critic_lstm=True, shared_lstm=False,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy", env_threaded, seed=0, verbose=0,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4, n_steps=64, batch_size=32, n_epochs=2,
        gamma=0.995, ent_coef=0.01, clip_range=0.1,
    )

    a_threaded  = _collect_actions(model, env_threaded,  threaded=True)
    a_stateless = _collect_actions(model, env_stateless, threaded=False)

    diverge_at = None
    for i, (x, y) in enumerate(zip(a_threaded, a_stateless)):
        if not np.allclose(x, y, atol=1e-7):
            diverge_at = i
            break

    if diverge_at is None:
        raise AssertionError(
            "LSTM threading FAILED: threaded and stateless action sequences "
            "are identical across all 10 steps. The LSTM hidden state is not "
            "being propagated; RecurrentPPO has degenerated to feedforward."
        )
    if diverge_at == 0:
        print(
            "WARNING: actions diverge at step 0. With both runs starting from "
            "zero hidden state and identical observations, step 0 should agree. "
            "Check seed propagation."
        )
    print(f"LSTM threading verified: actions diverge after step {diverge_at}")


if __name__ == "__main__":
    main()
