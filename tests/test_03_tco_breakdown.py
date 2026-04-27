"""Tests mimicking notebook 03_tco_breakdown."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use("Agg")

from dc_tco.config import load_config
from dc_tco.simulation import run_simulation
from dc_tco.plotting import (
    plot_tco_breakdown,
    plot_server_timeline,
    plot_stranded_power,
    plot_demand_curve,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')


def _run_baseline():
    cfg = load_config(CONFIG_PATH)
    return run_simulation(cfg)


def test_simulation_produces_tco():
    result = _run_baseline()
    assert result.total_tco > 0
    assert result.tco_breakdown.total_capex > 0
    assert result.tco_breakdown.total_opex > 0


def test_capex_plus_opex_equals_total():
    result = _run_baseline()
    expected = result.tco_breakdown.total_capex + result.tco_breakdown.total_opex
    assert abs(result.tco_breakdown.total - expected) < 1.0


def test_final_state_has_servers():
    result = _run_baseline()
    final = result.quarterly_states[-1]
    assert final.total_servers > 0
    assert final.total_racks > 0


def test_tco_by_year_covers_all_years():
    result = _run_baseline()
    cfg = load_config(CONFIG_PATH)
    assert len(result.tco_by_year) == cfg.simulation.total_years


def test_plot_tco_breakdown_returns_figure():
    result = _run_baseline()
    fig = plot_tco_breakdown(result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_server_timeline_returns_figure():
    result = _run_baseline()
    fig = plot_server_timeline(result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_demand_curve_returns_figure():
    result = _run_baseline()
    fig = plot_demand_curve(result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_stranded_power_returns_figure():
    result = _run_baseline()
    fig = plot_stranded_power(result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_batches_have_valid_fields():
    result = _run_baseline()
    assert len(result.all_batches) > 0
    for b in result.all_batches:
        assert b.num_servers > 0
        assert b.server.code_name
        assert b.start_q >= 0
        assert b.decom_q > b.start_q
