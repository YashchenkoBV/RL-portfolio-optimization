# portfolio_env.py
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Long-only portfolio environment.

    Conventions:
      - at time t (price index), observation uses data up to t
      - action picks weights w_t
      - portfolio is updated using next-day asset returns r_t = P[t+1]/P[t] - 1
      - then t <- t+1
    """

    def __init__(
        self,
        prices,                 # pd.DataFrame or path to csv
        window=50,              # feature lookback (in days of returns)
        trans_cost=0.0,         # wealth impact coefficient
        cost_model="trade_l1",  # "trade_0" | "trade_l1" | "trade_l2"
        turnover_penalty=0.0,   # default reward only (if reward_fn=None)
        reward_fn=None,         # callable(ctx) -> float
        feature_fn=None,        # callable(returns_window)->1d features (optional)
        start_idx=None,         # price index to start at (>= window)
        end_idx=None,           # price index to end at (<= len(prices)-1)
        episode_length=None,    # if set, end = start + episode_length
        random_start=False,     # if True, sample start each reset
        seed=None,
        initial_value=1.0,
    ):
        super().__init__()

        # Load prices
        if isinstance(prices, str):
            df = pd.read_csv(prices)
        else:
            df = prices.copy()

        # Handle dates
        if "Date" in df.columns:
            self.dates = df["Date"].to_numpy()
            df = df.drop(columns=["Date"])
        else:
            self.dates = np.arange(len(df))

        self.assets = list(df.columns)
        P = df.to_numpy(dtype=np.float64)  # (T, N)
        self.P = P
        self.T, self.N = P.shape

        # Precomputed simple and log returns, length T-1
        self.simple_ret = P[1:] / P[:-1] - 1.0
        self.log_ret = np.log(P[1:] / P[:-1])

        # Store parameters
        self.window = int(window)
        self.trans_cost = float(trans_cost)
        self.cost_model = cost_model
        self.turnover_penalty = float(turnover_penalty)
        self.reward_fn = reward_fn
        self.feature_fn = feature_fn
        self.episode_length = episode_length
        self.random_start = bool(random_start)
        self.initial_value = float(initial_value)

        self.base_start = self.window if start_idx is None else int(start_idx)
        self.base_end = (self.T - 1) if end_idx is None else int(end_idx)

        assert self.base_start >= self.window, "start_idx must be >= window"
        assert self.base_end <= self.T - 1, "end_idx must be <= last price index"
        assert self.base_end - self.base_start >= 1, "Need at least 1 step"

        self.rng = np.random.RandomState(seed)

        # Feature and observation spaces
        feat_dim = self._feature_dim()
        obs_dim = feat_dim + self.N + 1  # features + prev weights + drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.N,), dtype=np.float32
        )

        # state
        self.t = None
        self.start = None
        self.end = None
        self.value = None
        self.peak = None
        self.dd = None
        self.w_prev = None
        self.history = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.start, self.end = self._sample_episode_range()

        self.t = self.start
        self.value = self.initial_value
        self.peak = self.initial_value
        self.dd = 0.0
        self.w_prev = np.ones(self.N, dtype=np.float64) / self.N

        self.history = {
            "date": [],
            "value": [],
            "port_ret": [],
            "reward": [],
            "turnover": [],
            "cost": [],
            "dd": [],
        }

        obs = self._get_obs()
        info = {"t": self.t, "date": self.dates[self.t]}
        return obs, info

    def step(self, action):
        if self.t >= self.end:
            raise RuntimeError("Episode is over. Call reset().")

        w = self._softmax(action).astype(np.float64)
        r = self.simple_ret[self.t]  # returns for interval t -> t+1

        port_ret = float(np.dot(w, r))
        turnover = float(np.sum(np.abs(w - self.w_prev)))

        # Transaction costs
        if self.cost_model == "trade_0" or self.trans_cost <= 0:
            cost = 0.0
        elif self.cost_model == "trade_l1":
            cost = self.trans_cost * turnover
        else:  # cost_model == "trade_l2":
            cost = self.trans_cost * (turnover ** 2)
        cost_mult = max(0.0, 1.0 - cost)  # how cost contributes to wealth

        # Wealth update
        gross = 1.0 + port_ret
        if gross <= 0:
            gross = 1e-12
        v_prev = self.value
        v_next = v_prev * gross * cost_mult
        self.value = float(v_next)

        # Drawdown update
        if self.value > self.peak:
            self.peak = self.value
        self.dd = float(self.value / self.peak - 1.0)

        # REWARD SECTION
        log_growth = float(np.log(max(self.value / v_prev, 1e-12)))

        # determine whether the terminal bonus is needed
        terminated = (self.t + 1 >= self.end)

        # All we might need to calculate reward
        ctx = {
            "t": self.t,
            "date": self.dates[self.t + 1] if (self.t + 1) < len(self.dates) else self.dates[self.t],
            "weights": w,
            "weights_prev": self.w_prev,
            "asset_returns": r,
            "portfolio_return": port_ret,
            "turnover": turnover,
            "cost": cost,
            "value_prev": v_prev,
            "value": self.value,
            "peak": self.peak,
            "drawdown": self.dd,
            "log_growth": log_growth,
            "history": self.history,
            "is_terminal": terminated,
        }

        if self.reward_fn is None:
            reward = log_growth - self.turnover_penalty * turnover
        else:
            reward = float(self.reward_fn(ctx))

        # book-keeping (realized at next day)
        self.history["date"].append(ctx["date"])
        self.history["value"].append(self.value)
        self.history["port_ret"].append(port_ret)
        self.history["reward"].append(reward)
        self.history["turnover"].append(turnover)
        self.history["cost"].append(cost)
        self.history["dd"].append(self.dd)

        # advance time
        w_after = w * (1 + r)
        w_sum = w_after.sum()
        self.w_prev = w_after / w_sum if w_sum > 0 else w
        self.t += 1

        terminated = (self.t >= self.end)
        truncated = False

        obs = self._get_obs()
        info = {
            "t": self.t,
            "date": self.dates[self.t] if self.t < len(self.dates) else self.dates[-1],
            "value": self.value,
            "drawdown": self.dd,
            "turnover": turnover,
            "cost": cost,
            "portfolio_return": port_ret,
        }
        return obs, reward, terminated, truncated, info

    # ---------------- helpers ----------------

    def _sample_episode_range(self):
        start = self.base_start
        end = self.base_end

        if self.episode_length is not None:
            L = int(self.episode_length)
            max_start = self.base_end - L
            if max_start < self.base_start:
                raise ValueError("episode_length too long for chosen [start_idx, end_idx].")
            if self.random_start:
                start = int(self.rng.randint(self.base_start, max_start + 1))
            end = start + L

        return start, end

    def _feature_dim(self):
        if self.feature_fn is None:
            return self.window * self.N
        # probe once with dummy input
        dummy = np.zeros((self.window, self.N), dtype=np.float64)
        out = np.asarray(self.feature_fn(dummy)).reshape(-1)
        return int(out.shape[0])

    @staticmethod
    def _softmax(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x = x - np.max(x)
        e = np.exp(x)
        s = e.sum()
        return e / s

    def _cost_amount(self, turnover):
        if self.cost_model == "trade_0" or self.trans_cost <= 0:
            return 0.0
        if self.cost_model == "trade_l2":
            return self.trans_cost * (turnover ** 2)
        # default: linear in turnover
        return self.trans_cost * turnover

    def _get_obs(self):
        t = int(self.t)
        a = t - self.window
        b = t
        rwin = self.log_ret[a:b]  # (window, N)

        if self.feature_fn is None:
            feats = rwin.reshape(-1)
        else:
            feats = np.asarray(self.feature_fn(rwin)).reshape(-1)

        obs = np.concatenate([feats, self.w_prev, np.array([self.dd])], axis=0)
        return obs.astype(np.float32)

    # convenient for eval scripts
    def get_history_df(self):
        import pandas as pd
        return pd.DataFrame(self.history)
