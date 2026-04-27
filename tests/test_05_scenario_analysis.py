"""Tests mimicking notebook 05_scenario_analysis."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy

import matplotlib
matplotlib.use("Agg")
import numpy as np

from dc_tco.config import load_config
from dc_tco.simulation import run_simulation
from dc_tco.scenarios import run_monte_carlo, compare_policies_mc
from dc_tco.scenarios import _check_convergence
from dc_tco.plotting import plot_monte_carlo_distribution


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'scenarios')


def test_demand_shock_increases_tco() -> None:
    cfg_base = load_config(CONFIG_PATH)
    r_base = run_simulation(cfg_base)

    cfg_demand = load_config(CONFIG_PATH, os.path.join(SCENARIOS_DIR, 'demand_shock.yaml'))
    r_demand = run_simulation(cfg_demand)

    assert r_demand.total_tco > r_base.total_tco


def test_hw_capability_shock() -> None:
    cfg_base = load_config(CONFIG_PATH)
    r_base = run_simulation(cfg_base)

    cfg_hw = load_config(CONFIG_PATH, os.path.join(SCENARIOS_DIR, 'hw_capability_shock.yaml'))
    r_hw = run_simulation(cfg_hw)

    # HW capability shock changes fleet composition and TCO
    assert r_hw.total_tco != r_base.total_tco


def test_price_shock() -> None:
    cfg_base = load_config(CONFIG_PATH)
    r_base = run_simulation(cfg_base)

    cfg_price = load_config(CONFIG_PATH, os.path.join(SCENARIOS_DIR, 'price_shock.yaml'))
    r_price = run_simulation(cfg_price)

    # Price shock (cheaper HW) should reduce TCO
    assert r_price.total_tco < r_base.total_tco


def test_monte_carlo_basic() -> None:
    cfg = load_config(CONFIG_PATH)
    cfg_mc = copy.deepcopy(cfg)
    cfg_mc.monte_carlo.num_trials = 5

    mc_result = run_monte_carlo(cfg_mc, policy_name="baseline")

    assert mc_result.num_trials == 5
    assert mc_result.mean > 0
    assert mc_result.std >= 0
    assert mc_result.p5 <= mc_result.p50 <= mc_result.p95
    assert mc_result.ci_95[0] <= mc_result.ci_95[1]


def test_monte_carlo_stats_consistent() -> None:
    cfg = load_config(CONFIG_PATH)
    cfg_mc = copy.deepcopy(cfg)
    cfg_mc.monte_carlo.num_trials = 10
    cfg_mc.monte_carlo.seed = 42

    mc_result = run_monte_carlo(cfg_mc, policy_name="baseline")

    assert len(mc_result.tco_values) == 10
    assert mc_result.p5 >= min(mc_result.tco_values)
    assert mc_result.p95 <= max(mc_result.tco_values)


def test_plot_monte_carlo_distribution() -> None:
    cfg = load_config(CONFIG_PATH)
    cfg_mc = copy.deepcopy(cfg)
    cfg_mc.monte_carlo.num_trials = 5

    mc_result = run_monte_carlo(cfg_mc, policy_name="baseline")

    fig = plot_monte_carlo_distribution(mc_result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_compare_policies_mc() -> None:
    cfg = load_config(CONFIG_PATH)
    cfg_mc = copy.deepcopy(cfg)
    cfg_mc.monte_carlo.num_trials = 5

    mc_comparison = compare_policies_mc(
        cfg_mc,
        policy_names=["baseline", "replace_all"],
    )

    assert "baseline" in mc_comparison
    assert "replace_all" in mc_comparison
    for name, mc in mc_comparison.items():
        assert mc.num_trials == 5
        assert mc.mean > 0


# ---------------------------------------------------------------------------
# _check_convergence unit tests
# ---------------------------------------------------------------------------

class TestCheckConvergence:

    def test_few_samples_returns_true(self) -> None:
        """< 4000 samples always returns True (too few to check)."""
        values = np.ones(100)
        assert _check_convergence(values, threshold=0.01)

    def test_exactly_3999_returns_true(self) -> None:
        values = np.ones(3999)
        assert _check_convergence(values, threshold=0.01)

    def test_constant_values_converges(self) -> None:
        """4000+ identical values should converge."""
        values = np.full(5000, 1e9)
        assert _check_convergence(values, threshold=0.01)

    def test_stable_random_converges(self) -> None:
        """Large sample from a narrow distribution should converge."""
        rng = np.random.default_rng(42)
        values = rng.normal(loc=1e9, scale=1e6, size=5000)
        assert _check_convergence(values, threshold=0.05)

    def test_diverging_series_does_not_converge(self) -> None:
        """Monotonically increasing series should NOT converge with tight threshold."""
        values = np.linspace(1e9, 1e12, 5000)
        assert not _check_convergence(values, threshold=1e-6)

    def test_step_change_does_not_converge(self) -> None:
        """A sudden jump halfway should fail with a tight threshold."""
        values = np.concatenate([
            np.full(2500, 1e9),
            np.full(2500, 1e12),
        ])
        assert not _check_convergence(values, threshold=1e-6)

    def test_threshold_sensitivity(self) -> None:
        """Same data can converge or not depending on threshold."""
        rng = np.random.default_rng(99)
        values = rng.normal(loc=1e9, scale=1e8, size=5000)
        # Very tight threshold may fail
        result_tight = _check_convergence(values, threshold=1e-10)
        # Very loose threshold should pass
        result_loose = _check_convergence(values, threshold=1.0)
        assert result_loose
        # tight may or may not pass, but loose should always be >= tight
        assert result_loose or not result_tight
