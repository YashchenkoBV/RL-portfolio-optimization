"""Unit tests for CSI/CII in metrics.py."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "RL_optim")))
# fallback for CI / container where source lives at /mnt/project
sys.path.insert(0, "/mnt/project")

from metrics import compute_csi, compute_cii, to_prices_df


# ── helpers ──

def _csi(values):
    """Shorthand: compute CSI for a single series."""
    df = pd.DataFrame({"x": values})
    return float(compute_csi(df).iloc[0])


def _cii(values):
    df = pd.DataFrame({"x": values})
    return float(compute_cii(df).iloc[0])


# ── core CSI correctness ──

class TestComputeCsi:
    def test_perfectly_ascending(self):
        assert _csi([1, 2, 3, 4, 5]) == pytest.approx(100.0)

    def test_perfectly_descending(self):
        assert _csi([5, 4, 3, 2, 1]) == pytest.approx(0.0)

    def test_constant(self):
        assert _csi([7, 7, 7, 7]) == pytest.approx(100.0)

    def test_single_element(self):
        assert np.isnan(_csi([42]))  # n<2 → nan from metrics path
        # Note: rewards._csi_1d returns 100.0 for n<2; metrics returns nan.
        # Both are acceptable — metrics is for reporting, rewards needs a default.

    def test_two_up(self):
        assert _csi([1, 2]) == pytest.approx(100.0)

    def test_two_down(self):
        assert _csi([2, 1]) == pytest.approx(0.0)

    def test_two_equal(self):
        assert _csi([5, 5]) == pytest.approx(100.0)

    def test_v_shape(self):
        # [10, 5, 10] → s=[0, 0, 2], num=2, den=3 → 66.67%
        assert _csi([10, 5, 10]) == pytest.approx(200 / 3)

    def test_longer_ascending(self):
        assert _csi(list(range(1, 101))) == pytest.approx(100.0)

    def test_longer_descending(self):
        assert _csi(list(range(100, 0, -1))) == pytest.approx(0.0)


# ── CII = 100 - CSI ──

class TestComputeCii:
    def test_ascending_cii_zero(self):
        assert _cii([1, 2, 3, 4, 5]) == pytest.approx(0.0)

    def test_descending_cii_hundred(self):
        assert _cii([5, 4, 3, 2, 1]) == pytest.approx(100.0)

    def test_complement(self):
        vals = [3, 1, 4, 1, 5, 9, 2, 6]
        assert _csi(vals) + _cii(vals) == pytest.approx(100.0)


# ── multi-column support ──

class TestMultiColumn:
    def test_independent_columns(self):
        df = pd.DataFrame({
            "up": [1, 2, 3, 4, 5],
            "down": [5, 4, 3, 2, 1],
        })
        csi = compute_csi(df)
        assert csi["up"] == pytest.approx(100.0)
        assert csi["down"] == pytest.approx(0.0)


# ── consistency between metrics.py and rewards.py ──

class TestCrossModuleConsistency:
    """CSI from metrics.compute_csi must match rewards._csi_1d."""

    def test_same_result(self):
        from rewards import _csi_1d

        for values in [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [3, 3, 3],
            [10, 5, 10],
            [1, 3, 2, 4, 3, 5],
            list(np.random.RandomState(42).uniform(50, 100, size=60)),
        ]:
            csi_metrics = _csi(values)
            csi_rewards = _csi_1d(values)
            assert csi_metrics == pytest.approx(csi_rewards, abs=1e-10), \
                f"Mismatch on {values[:5]}..."


# ── Seagate docx sanity check ──

class TestSeagateEtalon:
    def test_denominator_matches_table(self):
        """Docx: 61 days + y0 = 62 points, etalon = 62*61/2 = 1891."""
        n = 62
        assert n * (n - 1) / 2 == 1891

    def test_reported_csi(self):
        """Docx: CSI = 1668/1891 * 100 = 88.21%."""
        assert pytest.approx(1668 / 1891 * 100, abs=0.01) == 88.21