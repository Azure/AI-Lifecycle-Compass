"""Tests for dc_tco.demand — workload demand generation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from dc_tco.config import ScalingMode, Config, load_config
from dc_tco.demand import build_demand_curve, compute_demand


CONFIG_PATH = "configs/default.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


# ---------------------------------------------------------------------------
# compute_demand — scaling modes
# ---------------------------------------------------------------------------

class TestScalingModes:

    def test_flat_demand_is_constant(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 5000.0
        assert compute_demand(0, cfg) == 5000.0
        assert compute_demand(10, cfg) == 5000.0
        assert compute_demand(59, cfg) == 5000.0

    def test_linear_demand_at_q0(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.LINEAR
        cfg.workload.initial_demand = 1000.0
        cfg.workload.scaling_factor = 1.1
        assert compute_demand(0, cfg) == pytest.approx(1000.0)

    def test_linear_demand_grows(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.LINEAR
        cfg.workload.initial_demand = 1000.0
        cfg.workload.scaling_factor = 1.1  # +10% per quarter
        # At q=10: 1000 * (1 + 0.1 * 10) = 2000
        assert compute_demand(10, cfg) == pytest.approx(2000.0)

    def test_exponential_demand_at_q0(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.EXPONENTIAL
        cfg.workload.initial_demand = 1000.0
        assert compute_demand(0, cfg) == pytest.approx(1000.0)

    def test_exponential_demand_grows(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.EXPONENTIAL
        cfg.workload.initial_demand = 1000.0
        cfg.workload.scaling_factor = 1.05
        # At q=4: 1000 * 1.05^4
        expected = 1000.0 * (1.05 ** 4)
        assert compute_demand(4, cfg) == pytest.approx(expected)

    def test_unknown_scaling_raises(self, cfg: Config) -> None:
        cfg.workload.scaling = "bogus"  # type: ignore
        with pytest.raises(ValueError, match="Unknown workload scaling mode"):
            compute_demand(0, cfg)


# ---------------------------------------------------------------------------
# compute_demand — demand shock scenario
# ---------------------------------------------------------------------------

class TestDemandShock:

    def test_shock_disabled_has_no_effect(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 1000.0
        cfg.scenarios.demand_shock.enabled = False
        assert compute_demand(40, cfg) == 1000.0

    def test_shock_before_shock_quarter_returns_baseline(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 1000.0
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 5  # q=20
        cfg.scenarios.demand_shock.alpha = 3.0
        assert compute_demand(19, cfg) == 1000.0

    def test_shock_flat_at_shock_quarter(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 1000.0
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 5
        cfg.scenarios.demand_shock.alpha = 3.0
        cfg.scenarios.demand_shock.post_shock_growth = 0.0
        # shock_baseline = 1000, shock_level = 3 * 1000 = 3000
        assert compute_demand(20, cfg) == pytest.approx(3000.0)

    def test_shock_exponential_at_shock_quarter(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.EXPONENTIAL
        cfg.workload.initial_demand = 1000.0
        cfg.workload.scaling_factor = 1.05
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 2  # q=8
        cfg.scenarios.demand_shock.alpha = 2.0
        cfg.scenarios.demand_shock.post_shock_growth = 0.0
        shock_baseline = 1000.0 * (1.05 ** 8)
        assert compute_demand(8, cfg) == pytest.approx(2.0 * shock_baseline)

    def test_shock_linear_at_shock_quarter(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.LINEAR
        cfg.workload.initial_demand = 1000.0
        cfg.workload.scaling_factor = 1.1
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 2  # q=8
        cfg.scenarios.demand_shock.alpha = 2.0
        cfg.scenarios.demand_shock.post_shock_growth = 0.0
        shock_baseline = 1000.0 * (1.0 + 0.1 * 8)
        assert compute_demand(8, cfg) == pytest.approx(2.0 * shock_baseline)

    def test_shock_plateau_no_growth(self, cfg: Config) -> None:
        """Post-shock demand stays flat when post_shock_growth=0."""
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 1000.0
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 5  # q=20
        cfg.scenarios.demand_shock.alpha = 3.0
        cfg.scenarios.demand_shock.post_shock_growth = 0.0
        val_q20 = compute_demand(20, cfg)
        val_q30 = compute_demand(30, cfg)
        assert val_q20 == pytest.approx(val_q30)

    def test_shock_with_post_shock_growth(self, cfg: Config) -> None:
        """Post-shock demand continues to grow when post_shock_growth > 0."""
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 1000.0
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 5  # q=20
        cfg.scenarios.demand_shock.alpha = 3.0
        cfg.scenarios.demand_shock.post_shock_growth = 0.1
        val_q20 = compute_demand(20, cfg)
        val_q24 = compute_demand(24, cfg)
        assert val_q24 > val_q20


# ---------------------------------------------------------------------------
# build_demand_curve
# ---------------------------------------------------------------------------

class TestBuildDemandCurve:

    def test_length_matches_num_quarters(self, cfg: Config) -> None:
        curve = build_demand_curve(cfg)
        assert len(curve) == cfg.simulation.num_quarters

    def test_returns_ndarray(self, cfg: Config) -> None:
        curve = build_demand_curve(cfg)
        assert isinstance(curve, np.ndarray)

    def test_all_positive(self, cfg: Config) -> None:
        curve = build_demand_curve(cfg)
        assert np.all(curve > 0)

    def test_exponential_curve_is_monotonically_increasing(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.EXPONENTIAL
        cfg.workload.scaling_factor = 1.05
        cfg.scenarios.demand_shock.enabled = False
        curve = build_demand_curve(cfg)
        assert np.all(np.diff(curve) > 0)

    def test_flat_curve_is_constant(self, cfg: Config) -> None:
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 42.0
        cfg.scenarios.demand_shock.enabled = False
        curve = build_demand_curve(cfg)
        assert np.all(curve == 42.0)

    def test_shock_visible_in_curve(self, cfg: Config) -> None:
        """Demand shock should cause a visible jump in the curve."""
        cfg.workload.scaling = ScalingMode.FLAT
        cfg.workload.initial_demand = 1000.0
        cfg.scenarios.demand_shock.enabled = True
        cfg.scenarios.demand_shock.shock_year = 2  # q=8
        cfg.scenarios.demand_shock.alpha = 5.0
        cfg.scenarios.demand_shock.post_shock_growth = 0.0
        curve = build_demand_curve(cfg)
        assert curve[7] == pytest.approx(1000.0)
        assert curve[8] == pytest.approx(5000.0)
