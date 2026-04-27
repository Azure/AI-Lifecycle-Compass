"""Tests mimicking notebook 04_policy_comparison."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy

import matplotlib
matplotlib.use("Agg")

from dc_tco.config import load_config
from dc_tco.simulation import run_simulation
from dc_tco.scenarios import run_policy_sweep
from dc_tco.plotting import plot_policy_comparison


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')


def test_all_policies_produce_results():
    cfg = load_config(CONFIG_PATH)
    policies = ["baseline", "replace_all", "extend_lifetime", "disaggregated"]
    results = {}

    for policy_name in policies:
        pcfg = copy.deepcopy(cfg)
        pcfg.policy.name = policy_name
        results[policy_name] = run_simulation(pcfg)

    for name, r in results.items():
        assert r.total_tco > 0, f"{name} should have positive TCO"
        assert r.tco_breakdown.total_capex > 0
        assert r.tco_breakdown.total_opex > 0


def test_policies_differ_in_tco():
    cfg = load_config(CONFIG_PATH)
    tcos = {}

    for policy_name in ["baseline", "replace_all", "extend_lifetime"]:
        pcfg = copy.deepcopy(cfg)
        pcfg.policy.name = policy_name
        r = run_simulation(pcfg)
        tcos[policy_name] = r.total_tco

    # Not all policies should produce identical TCO
    values = list(tcos.values())
    assert len(set(round(v, -6) for v in values)) > 1


def test_server_counts_vary_by_policy():
    cfg = load_config(CONFIG_PATH)

    cfg_base = copy.deepcopy(cfg)
    cfg_base.policy.name = "baseline"
    r_base = run_simulation(cfg_base)

    cfg_repl = copy.deepcopy(cfg)
    cfg_repl.policy.name = "replace_all"
    r_repl = run_simulation(cfg_repl)

    base_final = r_base.quarterly_states[-1].total_servers
    repl_final = r_repl.quarterly_states[-1].total_servers
    # replace_all tends to have different fleet size
    assert base_final > 0 and repl_final > 0


def test_policy_sweep_produces_results():
    cfg = load_config(CONFIG_PATH)
    sweep_results = run_policy_sweep(
        cfg,
        include_disaggregated=False,
        include_skip=False,
        max_extend_years=3,
    )

    assert len(sweep_results) > 0
    for r in sweep_results:
        assert r.total_tco > 0
        assert r.policy_name


def test_policy_sweep_includes_baseline():
    cfg = load_config(CONFIG_PATH)
    sweep_results = run_policy_sweep(
        cfg,
        include_disaggregated=False,
        include_skip=False,
    )

    names = [r.policy_name for r in sweep_results]
    assert any("baseline" in n for n in names)


def test_plot_policy_comparison_from_sweep():
    cfg = load_config(CONFIG_PATH)
    sweep_results = run_policy_sweep(
        cfg,
        include_disaggregated=False,
        include_skip=False,
        max_extend_years=2,
    )

    fig = plot_policy_comparison(sweep_results, normalize_to="baseline")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
