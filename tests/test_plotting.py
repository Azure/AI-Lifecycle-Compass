"""Tests for plotting module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np

from dc_tco.hardware import GpuSpec
from dc_tco.tco import TcoBreakdown
from dc_tco.simulation import QuarterState, SimulationResult
from dc_tco.scenarios import SweepResult, MonteCarloResult
from dc_tco.config import load_config
from dc_tco.plotting import (
    plot_gpu_trends,
    plot_server_timeline,
    plot_tco_breakdown,
    plot_stranded_power,
    plot_policy_comparison,
    plot_monte_carlo_distribution,
)


@pytest.fixture
def sample_roadmap():
    return [
        GpuSpec(code_name="GPU_A", release_quarter=8, tdp=250, tflops=30,
                mem_bw=900, mem_cap=16, net_bw=300, cost=10000),
        GpuSpec(code_name="GPU_B", release_quarter=16, tdp=300, tflops=60,
                mem_bw=1500, mem_cap=40, net_bw=300, cost=15000),
        GpuSpec(code_name="Future_C", release_quarter=24, tdp=350, tflops=100,
                mem_bw=2000, mem_cap=80, net_bw=300, cost=20000),
    ]


@pytest.fixture
def sample_tco():
    return TcoBreakdown(
        capex_server=1e6, capex_rack=2e5,
        capex_power_provisioning=1e5, capex_cooling_provisioning=5e4,
        capex_network=3e4,
        opex_energy=5e5, opex_cooling=1e5,
        opex_maintenance=2e5, opex_network=1e4,
    )


@pytest.fixture
def sample_quarter_state(sample_tco):
    return QuarterState(
        quarter=0, year=0, demand=1000.0, model_size_B=7.0,
        total_servers=10, total_racks=2, total_capacity=2000.0,
        shortfall=0.0, servers_added=10, servers_decommissioned=0,
        stranded_power_kw=50.0, tco=sample_tco,
        gen_server_counts={"GPU_A": 6, "GPU_B": 4},
    )


@pytest.fixture
def sample_sim_result(sample_quarter_state, sample_tco):
    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'))
    states = [sample_quarter_state]
    for q in range(1, 4):
        s = QuarterState(
            quarter=q, year=q // 4, demand=1000.0 + q * 100,
            model_size_B=7.0, total_servers=10 + q, total_racks=2,
            total_capacity=2000.0, shortfall=0.0, servers_added=1,
            servers_decommissioned=0, stranded_power_kw=50.0 - q * 5,
            tco=sample_tco,
            gen_server_counts={"GPU_A": 6, "GPU_B": 4 + q},
        )
        states.append(s)
    return SimulationResult(
        quarterly_states=states,
        all_batches=[],
        total_tco=sample_tco.total * 4,
        tco_by_year={0: sample_tco},
        tco_breakdown=sample_tco,
        config=cfg,
    )


@pytest.fixture
def sample_sweep_results():
    return [
        SweepResult(policy_name="baseline", variant="regular",
                    total_tco=1e9, tco_breakdown={}, skip_gens=[], extend_years=0),
        SweepResult(policy_name="replace_all", variant="regular",
                    total_tco=0.9e9, tco_breakdown={}, skip_gens=[], extend_years=0),
        SweepResult(policy_name="extend_2y", variant="regular",
                    total_tco=1.1e9, tco_breakdown={}, skip_gens=[], extend_years=2),
    ]


@pytest.fixture
def sample_mc_result():
    rng = np.random.default_rng(42)
    values = rng.normal(1e9, 1e8, 100)
    return MonteCarloResult(
        num_trials=100,
        tco_values=values,
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        ci_95=(float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))),
        p5=float(np.percentile(values, 5)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
        converged=True,
        policy_name="baseline",
    )


def test_plot_gpu_trends(sample_roadmap):
    fig = plot_gpu_trends(sample_roadmap)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_gpu_trends_subset(sample_roadmap):
    fig = plot_gpu_trends(sample_roadmap, metrics=["tflops", "cost"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_server_timeline(sample_sim_result):
    fig = plot_server_timeline(sample_sim_result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_tco_breakdown(sample_sim_result):
    fig = plot_tco_breakdown(sample_sim_result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_stranded_power(sample_sim_result):
    fig = plot_stranded_power(sample_sim_result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_policy_comparison(sample_sweep_results):
    fig = plot_policy_comparison(sample_sweep_results)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_policy_comparison_absolute(sample_sweep_results):
    fig = plot_policy_comparison(sample_sweep_results, normalize_to=None)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_policy_comparison_no_match(sample_sweep_results):
    fig = plot_policy_comparison(sample_sweep_results, variant="disaggregated")
    assert fig is None


def test_plot_monte_carlo_distribution(sample_mc_result):
    fig = plot_monte_carlo_distribution(sample_mc_result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
