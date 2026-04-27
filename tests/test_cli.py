"""Tests for CLI module."""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import patch
from io import StringIO

from dc_tco.cli import main


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'scenarios')


def test_no_command_prints_help():
    """main() with no arguments prints help and returns."""
    with patch('sys.stdout', new_callable=StringIO) as mock_out:
        main([])
    output = mock_out.getvalue()
    assert "dc-tco" in output or "usage" in output.lower()


def test_run_command(tmp_path):
    result_dir = str(tmp_path / "results")
    main(["run", "--config", CONFIG_PATH, "--output", result_dir])
    assert (tmp_path / "results" / "summary.json").exists()
    assert (tmp_path / "results" / "quarterly_states.csv").exists()


def test_run_command_with_policy(tmp_path):
    result_dir = str(tmp_path / "results")
    main(["run", "--config", CONFIG_PATH, "--policy", "replace_all",
          "--output", result_dir])
    with open(tmp_path / "results" / "summary.json") as f:
        summary = json.load(f)
    assert summary["policy"] == "replace_all"


def test_run_command_no_output(capsys):
    """run without --output just prints to stdout."""
    main(["run", "--config", CONFIG_PATH])
    captured = capsys.readouterr()
    assert "Total TCO" in captured.out


def test_run_command_with_scenario(tmp_path, capsys):
    """run with --scenario loads the scenario overlay."""
    scenario = os.path.join(SCENARIOS_DIR, "demand_shock.yaml")
    result_dir = str(tmp_path / "results")
    main(["run", "--config", CONFIG_PATH, "--scenario", scenario,
          "--output", result_dir])
    captured = capsys.readouterr()
    assert "Total TCO" in captured.out
    assert (tmp_path / "results" / "summary.json").exists()


def test_run_csv_has_expected_columns(tmp_path):
    """quarterly_states.csv should contain key columns."""
    result_dir = str(tmp_path / "results")
    main(["run", "--config", CONFIG_PATH, "--output", result_dir])
    import pandas as pd
    df = pd.read_csv(tmp_path / "results" / "quarterly_states.csv")
    for col in ["quarter", "demand", "total_servers", "tco_total"]:
        assert col in df.columns


def test_sweep_command(tmp_path):
    result_dir = str(tmp_path / "results")
    main(["sweep", "--config", CONFIG_PATH, "--output", result_dir,
          "--no-disagg", "--no-skip"])
    assert (tmp_path / "results" / "policy_sweep.csv").exists()


def test_sweep_prints_table(capsys):
    main(["sweep", "--config", CONFIG_PATH, "--no-disagg", "--no-skip"])
    captured = capsys.readouterr()
    assert "baseline" in captured.out


def test_sweep_csv_has_expected_columns(tmp_path):
    """policy_sweep.csv should contain policy and tco columns."""
    result_dir = str(tmp_path / "results")
    main(["sweep", "--config", CONFIG_PATH, "--output", result_dir,
          "--no-disagg", "--no-skip"])
    import pandas as pd
    df = pd.read_csv(tmp_path / "results" / "policy_sweep.csv")
    assert "policy" in df.columns
    assert "total_tco" in df.columns


def test_monte_carlo_command(tmp_path):
    result_dir = str(tmp_path / "results")
    main(["monte-carlo", "--config", CONFIG_PATH, "--trials", "5",
          "--seed", "42", "--output", result_dir])
    assert (tmp_path / "results" / "mc_tco_values.npy").exists()


def test_monte_carlo_prints_stats(capsys):
    main(["monte-carlo", "--config", CONFIG_PATH, "--trials", "5",
          "--seed", "42"])
    captured = capsys.readouterr()
    assert "Monte Carlo Results" in captured.out
    assert "Mean TCO" in captured.out


def test_monte_carlo_with_policy_override(capsys):
    """--policy flag should override the default policy."""
    main(["monte-carlo", "--config", CONFIG_PATH, "--trials", "3",
          "--seed", "1", "--policy", "replace_all"])
    captured = capsys.readouterr()
    assert "policy=replace_all" in captured.out


# ---------------------------------------------------------------------------
# rental command
# ---------------------------------------------------------------------------

def test_rental_prints_summary(capsys):
    """rental command prints the monthly breakdown."""
    main(["rental", "--config", CONFIG_PATH])
    captured = capsys.readouterr()
    assert "Rental TCO Summary" in captured.out
    assert "GPU compute" in captured.out
    assert "Goodput" in captured.out


def test_rental_with_overrides(capsys):
    """--months and --gpus flags override config."""
    main(["rental", "--config", CONFIG_PATH, "--months", "6", "--gpus", "64"])
    captured = capsys.readouterr()
    assert "64 GPUs" in captured.out
    assert "6 months" in captured.out


def test_rental_saves_json(tmp_path):
    """rental --output writes rental_summary.json."""
    result_dir = str(tmp_path / "results")
    main(["rental", "--config", CONFIG_PATH, "--output", result_dir])
    summary_path = tmp_path / "results" / "rental_summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        data = json.load(f)
    assert "monthly_total" in data
    assert "contract_total" in data
    assert "monthly_breakdown" in data
    assert "gpu" in data["monthly_breakdown"]
