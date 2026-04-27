"""Tests for simulation module (integration tests)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import pytest
from dc_tco.config import load_config
from dc_tco.simulation import run_simulation


@pytest.fixture
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    return load_config(config_path)


def test_baseline_simulation_completes(cfg):
    result = run_simulation(cfg)
    assert result.total_tco > 0
    assert len(result.quarterly_states) == cfg.simulation.num_quarters
    assert len(result.all_batches) > 0


def test_baseline_tco_breakdown(cfg):
    result = run_simulation(cfg)
    assert result.tco_breakdown.total_capex > 0
    assert result.tco_breakdown.total_opex > 0
    # Total should equal sum of parts
    expected = result.tco_breakdown.total_capex + result.tco_breakdown.total_opex
    assert abs(result.total_tco - expected) < 1.0


def test_demand_increases_over_time(cfg):
    result = run_simulation(cfg)
    first = result.quarterly_states[0].demand
    last = result.quarterly_states[-1].demand
    assert last > first


def test_replace_all_policy(cfg):
    cfg_ra = copy.deepcopy(cfg)
    cfg_ra.policy.name = "replace_all"
    result = run_simulation(cfg_ra)
    assert result.total_tco > 0
    assert len(result.quarterly_states) == cfg.simulation.num_quarters


def test_extend_lifetime_policy(cfg):
    cfg_ext = copy.deepcopy(cfg)
    cfg_ext.policy.name = "extend_lifetime"
    cfg_ext.policy.extend_lifetime_years = 2
    result = run_simulation(cfg_ext)
    assert result.total_tco > 0


def test_disaggregated_policy(cfg):
    cfg_dis = copy.deepcopy(cfg)
    cfg_dis.policy.name = "disaggregated"
    result = run_simulation(cfg_dis)
    assert result.total_tco > 0


def test_skip_generation(cfg):
    cfg_skip = copy.deepcopy(cfg)
    cfg_skip.policy.skip_generations = ["H100"]
    result = run_simulation(cfg_skip)
    # No batch should use H100
    for b in result.all_batches:
        assert b.server.code_name != "H100"


def test_scenario_override(cfg):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    scenario_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'scenarios', 'demand_shock.yaml')
    cfg_shock = load_config(config_path, scenario_path)
    assert cfg_shock.scenarios.demand_shock.enabled is True
    result = run_simulation(cfg_shock)
    assert result.total_tco > 0


def test_tco_by_year_sums_to_total(cfg):
    result = run_simulation(cfg)
    year_total = sum(t.total for t in result.tco_by_year.values())
    assert abs(year_total - result.total_tco) < 1.0


def test_quarterly_states_have_gen_counts(cfg):
    result = run_simulation(cfg)
    # At least some quarters should have servers
    has_servers = any(s.total_servers > 0 for s in result.quarterly_states)
    assert has_servers
    # Gen counts should match total count
    for s in result.quarterly_states:
        gen_total = sum(s.gen_server_counts.values())
        assert gen_total == s.total_servers


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------

def test_model_size_contraction_reduces_tco(cfg):
    """Scenario 2: model size contraction should reduce TCO vs baseline."""
    baseline = run_simulation(cfg)

    cfg_s2 = copy.deepcopy(cfg)
    cfg_s2.scenarios.model_size_contraction.enabled = True
    cfg_s2.scenarios.model_size_contraction.contraction_year = 5
    cfg_s2.scenarios.model_size_contraction.beta = 0.8
    result = run_simulation(cfg_s2)

    assert result.total_tco < baseline.total_tco


def test_hw_capability_shock_changes_tco(cfg):
    """Scenario 3: HW capability shock should change TCO vs baseline."""
    baseline = run_simulation(cfg)

    cfg_s3 = copy.deepcopy(cfg)
    cfg_s3.scenarios.hw_capability_shock.enabled = True
    cfg_s3.scenarios.hw_capability_shock.shock_year = 5
    cfg_s3.scenarios.hw_capability_shock.gamma = 3.0
    cfg_s3.scenarios.hw_capability_shock.epsilon = 0.05
    result = run_simulation(cfg_s3)

    assert result.total_tco != baseline.total_tco


def test_price_shock_reduces_tco(cfg):
    """Scenario 4: price shock should reduce TCO vs baseline."""
    baseline = run_simulation(cfg)

    cfg_s4 = copy.deepcopy(cfg)
    cfg_s4.scenarios.price_shock.enabled = True
    cfg_s4.scenarios.price_shock.shock_year = 5
    cfg_s4.scenarios.price_shock.delta = 0.6
    cfg_s4.scenarios.price_shock.eta = 0.10
    result = run_simulation(cfg_s4)

    assert result.total_tco < baseline.total_tco


def test_demand_shock_increases_tco(cfg):
    """Scenario 1: demand shock should increase TCO vs baseline."""
    baseline = run_simulation(cfg)

    cfg_s1 = copy.deepcopy(cfg)
    cfg_s1.scenarios.demand_shock.enabled = True
    cfg_s1.scenarios.demand_shock.shock_year = 5
    cfg_s1.scenarios.demand_shock.alpha = 3.0
    cfg_s1.scenarios.demand_shock.post_shock_growth = 0.0
    result = run_simulation(cfg_s1)

    assert result.total_tco > baseline.total_tco
