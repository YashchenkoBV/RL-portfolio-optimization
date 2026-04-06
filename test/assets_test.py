"""
Tests for Phase 1.1: cash asset in PortfolioEnv.

Covers:
  - Action/observation space dimensions include cash slot
  - Pure cash allocation earns exactly rf_daily
  - Mixed cash/equity allocations behave correctly
  - Weight drift after market moves includes cash component
  - Turnover accounts for cash rebalancing
  - Cash weight appears in observation vector
  - Initial weights start with zero cash
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "RL_optim")))
# fallback for CI / container where source lives at /mnt/project
sys.path.insert(0, "/mnt/project")

from portfolio_env import PortfolioEnv


# ─────────────────── fixtures ───────────────────


@pytest.fixture
def prices_3():
    """100 days, 3 assets, gentle random walk starting at 50."""
    rng = np.random.RandomState(42)
    P = 50.0 + np.cumsum(rng.randn(100, 3) * 0.3, axis=0)
    return pd.DataFrame(P, columns=["A", "B", "C"])


@pytest.fixture
def prices_1():
    """100 days, 1 asset — edge case for dimension handling."""
    rng = np.random.RandomState(7)
    P = 100.0 + np.cumsum(rng.randn(100) * 0.5)
    return pd.DataFrame(P, columns=["X"])


@pytest.fixture
def env(prices_3):
    return PortfolioEnv(prices_3, window=10, seed=0)


@pytest.fixture
def env_rf(prices_3):
    return PortfolioEnv(prices_3, window=10, rf_rate=0.05, seed=0)


@pytest.fixture
def env_cost(prices_3):
    return PortfolioEnv(prices_3, window=10, trans_cost=0.001, cost_model="trade_l1", seed=0)


@pytest.fixture
def env_1asset(prices_1):
    return PortfolioEnv(prices_1, window=10, rf_rate=0.03, seed=0)


# ─────────────────── dimensions ───────────────────


class TestCashDimensions:
    def test_action_space_is_n_plus_1(self, env):
        assert env.action_space.shape == (env.N + 1,)

    def test_obs_space_accounts_for_cash_weight(self, env):
        feat_dim = env.window * env.N
        expected = feat_dim + (env.N + 1) + 1  # features + w_prev(N+1) + dd
        assert env.observation_space.shape == (expected,)

    def test_single_asset_action_space(self, env_1asset):
        # 1 asset + 1 cash = 2
        assert env_1asset.action_space.shape == (2,)

    def test_single_asset_obs_space(self, env_1asset):
        feat_dim = env_1asset.window * 1
        expected = feat_dim + 2 + 1  # features + w_prev(1+1) + dd
        assert env_1asset.observation_space.shape == (expected,)

    def test_obs_shape_after_reset(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_obs_shape_after_step(self, env):
        env.reset()
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape


# ─────────────────── initial weights ───────────────────


class TestCashInitialWeights:
    def test_initial_cash_weight_is_zero(self, env):
        env.reset()
        assert env.w_prev[-1] == 0.0

    def test_initial_asset_weights_are_equal(self, env):
        env.reset()
        expected = np.ones(env.N) / env.N
        np.testing.assert_allclose(env.w_prev[:-1], expected)

    def test_initial_weights_sum_to_one(self, env):
        env.reset()
        # assets sum to 1, cash is 0 — total is 1
        assert env.w_prev.sum() == pytest.approx(1.0)

    def test_initial_cash_in_obs(self, env):
        obs, _ = env.reset()
        # w_prev is embedded after features, before drawdown
        feat_dim = env.window * env.N
        w_in_obs = obs[feat_dim : feat_dim + env.N + 1]
        assert w_in_obs[-1] == pytest.approx(0.0)


# ─────────────────── pure cash return ───────────────────


def _all_cash_action(n_assets):
    """Logits that push softmax to ~100% cash."""
    a = np.full(n_assets + 1, -10.0)
    a[-1] = 10.0
    return a


class TestPureCashReturn:
    def test_earns_rf_daily(self, env_rf):
        env_rf.reset()
        _, _, _, _, info = env_rf.step(_all_cash_action(env_rf.N))
        assert info["portfolio_return"] == pytest.approx(0.05 / 252, abs=1e-10)

    def test_value_compounds_over_steps(self, env_rf):
        env_rf.reset()
        action = _all_cash_action(env_rf.N)
        n = 20
        for _ in range(n):
            _, _, term, _, info = env_rf.step(action)
            if term:
                break
        expected = (1.0 + 0.05 / 252) ** n
        assert info["value"] == pytest.approx(expected, rel=1e-9)

    def test_zero_rf_earns_nothing(self, env):
        env.reset()
        _, _, _, _, info = env.step(_all_cash_action(env.N))
        assert info["portfolio_return"] == pytest.approx(0.0, abs=1e-9)

    def test_no_drawdown_in_pure_cash(self, env_rf):
        env_rf.reset()
        action = _all_cash_action(env_rf.N)
        for _ in range(30):
            _, _, term, _, info = env_rf.step(action)
            assert info["drawdown"] == pytest.approx(0.0)
            if term:
                break

    def test_single_asset_pure_cash(self, env_1asset):
        env_1asset.reset()
        action = np.array([-10.0, 10.0])
        _, _, _, _, info = env_1asset.step(action)
        assert info["portfolio_return"] == pytest.approx(0.03 / 252, abs=1e-10)


# ─────────────────── pure equity (zero cash) ───────────────────


def _no_cash_action(n_assets):
    """Logits that push softmax to ~0% cash."""
    a = np.zeros(n_assets + 1)
    a[-1] = -10.0
    return a


class TestPureEquityReturn:
    def test_matches_equal_weight_return(self, prices_3, env):
        """With near-zero cash, portfolio return should match
        a manual dot(1/N, r) calculation."""
        env.reset()
        t = env.t
        r = prices_3.values[t + 1] / prices_3.values[t] - 1.0
        expected = float(np.dot(np.ones(3) / 3, r))

        action = _no_cash_action(env.N)
        _, _, _, _, info = env.step(action)

        # softmax still gives ~exp(-10)/(3+exp(-10)) ≈ 1.5e-5 to cash
        assert info["portfolio_return"] == pytest.approx(expected, abs=1e-4)


# ─────────────────── mixed allocation ───────────────────


class TestMixedAllocation:
    def test_half_cash_half_equity(self, env_rf):
        """Equal logits → softmax gives 1/(N+1) to each slot including cash."""
        env_rf.reset()
        action = np.zeros(env_rf.N + 1)
        _, _, _, _, info = env_rf.step(action)

        # portfolio return should be between pure-equity and pure-cash
        rf_daily = 0.05 / 252
        t = env_rf.t - 1  # step already advanced
        r = env_rf.simple_ret[t]
        equity_ret = float(np.dot(np.ones(env_rf.N) / (env_rf.N + 1), r))
        cash_ret = rf_daily / (env_rf.N + 1)
        expected = equity_ret + cash_ret

        assert info["portfolio_return"] == pytest.approx(expected, rel=1e-8)


# ─────────────────── weight drift with cash ───────────────────


class TestCashWeightDrift:
    def test_w_prev_has_n_plus_1_elements(self, env):
        env.reset()
        env.step(np.zeros(env.N + 1))
        assert len(env.w_prev) == env.N + 1

    def test_w_prev_sums_to_one(self, env):
        env.reset()
        env.step(np.zeros(env.N + 1))
        assert env.w_prev.sum() == pytest.approx(1.0, abs=1e-12)

    def test_cash_weight_drifts(self, env_rf):
        """After one step, cash weight should differ from the softmax target
        because asset values moved but cash grew at rf."""
        env_rf.reset()
        action = np.zeros(env_rf.N + 1)  # target ≈ 1/(N+1) each
        target_cash = 1.0 / (env_rf.N + 1)
        env_rf.step(action)
        assert env_rf.w_prev[-1] != pytest.approx(target_cash, abs=1e-6)

    def test_drift_reflected_in_obs(self, env):
        env.reset()
        env.step(np.zeros(env.N + 1))
        obs, *_ = env.step(np.zeros(env.N + 1))

        feat_dim = env.window * env.N
        w_in_obs = obs[feat_dim : feat_dim + env.N + 1]

        # should match env.w_prev from the previous step's drift
        # (w_prev was set at end of step 1, obs at start of step 2 reads it)
        np.testing.assert_allclose(w_in_obs, env.w_prev, atol=1e-6)

    def test_pure_cash_drift_stays_pure(self, env_rf):
        """If we go 100% cash, after drift we should still be ~100% cash
        (no assets to drift away from)."""
        env_rf.reset()
        action = _all_cash_action(env_rf.N)
        env_rf.step(action)
        assert env_rf.w_prev[-1] > 0.999


# ─────────────────── turnover with cash ───────────────────


class TestCashTurnover:
    def test_initial_to_cash_causes_full_turnover(self, env_cost):
        """Going from [1/3, 1/3, 1/3, 0] to [~0, ~0, ~0, ~1] is ~2.0 turnover."""
        env_cost.reset()
        _, _, _, _, info = env_cost.step(_all_cash_action(env_cost.N))
        # L1: sum |0 - 1/3|*3 + |1 - 0| = 1 + 1 = 2
        assert info["turnover"] == pytest.approx(2.0, abs=0.01)

    def test_staying_in_cash_has_near_zero_turnover(self, env_cost):
        """After moving to cash, repeating the same action should produce
        near-zero turnover since drifted cash stays at ~100%."""
        env_cost.reset()
        action = _all_cash_action(env_cost.N)
        env_cost.step(action)  # first step: big turnover
        _, _, _, _, info = env_cost.step(action)  # second step
        assert info["turnover"] < 0.001

    def test_equity_to_equity_turnover_from_drift(self, env_cost):
        """Fixed equal-weight action should have small turnover after
        step 0 due to drift only."""
        env_cost.reset()
        action = _no_cash_action(env_cost.N)
        env_cost.step(action)  # step 0
        turnovers = []
        for _ in range(10):
            _, _, term, _, info = env_cost.step(action)
            turnovers.append(info["turnover"])
            if term:
                break
        assert all(0 < t < 0.15 for t in turnovers)

    def test_cost_paid_on_initial_cash_shift(self, env_cost):
        env_cost.reset()
        _, _, _, _, info = env_cost.step(_all_cash_action(env_cost.N))
        # turnover ≈ 2.0, cost = 0.001 * 2.0 = 0.002
        assert info["cost"] == pytest.approx(0.002, abs=0.001)


# ─────────────────── ctx passed to reward_fn ───────────────────


class TestCashInRewardContext:
    def test_ctx_weights_include_cash(self, prices_3):
        captured = {}

        def spy_reward(ctx):
            captured.update(ctx)
            return ctx["log_growth"]

        env = PortfolioEnv(prices_3, window=10, reward_fn=spy_reward, seed=0)
        env.reset()
        env.step(np.zeros(env.N + 1))

        assert len(captured["weights"]) == env.N + 1
        assert len(captured["weights_prev"]) == env.N + 1
        assert captured["weights"].sum() == pytest.approx(1.0)

    def test_ctx_log_growth_reflects_cash(self, prices_3):
        captured = {}

        def spy_reward(ctx):
            captured.update(ctx)
            return ctx["log_growth"]

        env = PortfolioEnv(prices_3, window=10, rf_rate=0.05, reward_fn=spy_reward, seed=0)
        env.reset()
        env.step(_all_cash_action(3))

        expected_lg = np.log(1.0 + 0.05 / 252)
        assert captured["log_growth"] == pytest.approx(expected_lg, abs=1e-9)