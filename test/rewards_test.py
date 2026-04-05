"""Unit tests for CSI-related logic in rewards.py."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "RL_optim")))
# fallback for CI / container where source lives at /mnt/project
sys.path.insert(0, "/mnt/project")

from rewards import (
    _csi_1d,
    _get_values,
    _csi_from_ctx,
    reward_log_growth_plus_delta_csi,
    reward_terminal_csi_bonus,
    reward_log_growth_minus_cii,
)


# ── helpers ──

def _make_ctx(values, current_value, log_growth=0.001, turnover=0.0,
              is_terminal=False):
    """Minimal ctx dict mimicking what PortfolioEnv passes to reward fns."""
    return {
        "value": current_value,
        "log_growth": log_growth,
        "turnover": turnover,
        "drawdown": 0.0,
        "is_terminal": is_terminal,
        "history": {"value": list(values)},
    }


# ── _csi_1d: core formula ──

class TestCsi1d:
    def test_perfectly_ascending(self):
        assert _csi_1d([1, 2, 3, 4, 5]) == pytest.approx(100.0)

    def test_perfectly_descending(self):
        assert _csi_1d([5, 4, 3, 2, 1]) == pytest.approx(0.0)

    def test_constant(self):
        # all values equal → every y_i >= y_j, so CSI = 100%
        assert _csi_1d([3, 3, 3, 3, 3]) == pytest.approx(100.0)

    def test_single_element(self):
        assert _csi_1d([42]) == pytest.approx(100.0)

    def test_two_elements_up(self):
        assert _csi_1d([1, 2]) == pytest.approx(100.0)

    def test_two_elements_down(self):
        assert _csi_1d([2, 1]) == pytest.approx(0.0)

    def test_two_elements_equal(self):
        assert _csi_1d([5, 5]) == pytest.approx(100.0)

    def test_v_shape(self):
        # [10, 5, 10]: s0=0, s1=0(5<10), s2=1(10>=5, but 10>=10→+1)=1
        # wait let me recalculate:
        # s0 = 0 (no predecessors counted)
        # s1: y1>=y0? 5>=10? no → s1 = 0
        # s2: y2>=y0? 10>=10? yes. y2>=y1? 10>=5? yes → s2 = 2
        # num = 0+0+2 = 2, den = 3*2/2 = 3 → 66.67%
        assert _csi_1d([10, 5, 10]) == pytest.approx(200 / 3)

    def test_symmetry_not_assumed(self):
        # [1, 3, 2] vs [1, 2, 3] should differ
        assert _csi_1d([1, 2, 3]) > _csi_1d([1, 3, 2])

    def test_seagate_etalon(self):
        """Docx Table 2.1: 61 trading days + implicit y0 → n=62, etalon=1891."""
        n = 62
        expected_den = n * (n - 1) / 2
        assert expected_den == 1891


# ── _get_values: include_current contract ──

class TestGetValues:
    def test_include_current_appends(self):
        ctx = _make_ctx([1.0, 2.0, 3.0], current_value=4.0)
        vals = _get_values(ctx, include_current=True)
        assert list(vals) == [1.0, 2.0, 3.0, 4.0]

    def test_exclude_current(self):
        ctx = _make_ctx([1.0, 2.0, 3.0], current_value=4.0)
        vals = _get_values(ctx, include_current=False)
        assert list(vals) == [1.0, 2.0, 3.0]

    def test_duplicate_value_still_appended(self):
        """Issue 1 fix: equal consecutive values must not be suppressed."""
        ctx = _make_ctx([1.0, 2.0, 3.0], current_value=3.0)
        vals = _get_values(ctx, include_current=True)
        assert len(vals) == 4
        assert vals[-1] == 3.0 and vals[-2] == 3.0

    def test_window_truncation(self):
        ctx = _make_ctx([1, 2, 3, 4, 5], current_value=6)
        vals = _get_values(ctx, include_current=True, window=3)
        assert list(vals) == [4, 5, 6]


# ── delta CSI reward: warmup guard ──

class TestDeltaCsiWarmup:
    def test_warmup_returns_log_growth_only(self):
        """Issue 2 fix: when history < W, delta CSI is undefined → fall back."""
        ctx = _make_ctx([1.0, 1.01], current_value=1.02, log_growth=0.005)
        r = reward_log_growth_plus_delta_csi(ctx, W=60, alpha=1.0, eta=0.0)
        assert r == pytest.approx(0.005)

    def test_full_window_includes_csi_component(self):
        """Once window is full, alpha * delta should contribute."""
        # History: 10 values with a dip. Current value recovers above all.
        # include_current=True drops the oldest and adds current → different CSI
        # than include_current=False which keeps the original window.
        history = [10, 11, 12, 8, 9, 10, 11, 12, 13, 14]  # dip at index 3
        ctx = _make_ctx(history, current_value=15.0, log_growth=0.001)
        r_with = reward_log_growth_plus_delta_csi(ctx, W=10, alpha=1.0, eta=0.0)
        r_without = reward_log_growth_plus_delta_csi(ctx, W=10, alpha=0.0, eta=0.0)
        # Adding 15 (above all) and dropping 10 (lowest) should improve CSI
        assert r_with > r_without


# ── terminal CSI bonus ──

class TestTerminalBonus:
    def test_bonus_fires_on_terminal(self):
        ascending = list(np.linspace(1.0, 2.0, 10))
        ctx = _make_ctx(ascending, current_value=2.1, log_growth=0.001,
                        is_terminal=True)
        r_term = reward_terminal_csi_bonus(ctx, W=None, bonus=1.0)
        ctx["is_terminal"] = False
        r_step = reward_terminal_csi_bonus(ctx, W=None, bonus=1.0)
        assert r_term > r_step

    def test_no_bonus_when_not_terminal(self):
        ctx = _make_ctx([1, 2, 3], current_value=4, log_growth=0.01,
                        is_terminal=False)
        r = reward_terminal_csi_bonus(ctx, W=None, bonus=10.0)
        assert r == pytest.approx(0.01)


# ── CII = 100 - CSI in reward context ──

class TestCiiReward:
    def test_ascending_no_penalty(self):
        ascending = list(np.linspace(1.0, 2.0, 59))
        ctx = _make_ctx(ascending, current_value=2.01, log_growth=0.001)
        r = reward_log_growth_minus_cii(ctx, W=60, lam=1.0, eta=0.0)
        # CII ≈ 0 for ascending series → reward ≈ log_growth
        assert r == pytest.approx(0.001, abs=0.01)

    def test_descending_strong_penalty(self):
        descending = list(np.linspace(2.0, 1.0, 59))
        ctx = _make_ctx(descending, current_value=0.99, log_growth=0.001)
        r = reward_log_growth_minus_cii(ctx, W=60, lam=1.0, eta=0.0)
        # CII ≈ 100 for descending → reward ≈ 0.001 - 1.0
        assert r < -0.5