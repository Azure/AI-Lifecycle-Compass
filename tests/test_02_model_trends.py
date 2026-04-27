"""Tests mimicking notebook 02_model_trends."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from dc_tco.config import load_config, ProjectionMode
from dc_tco.models import (
    load_notable_models,
    compute_yearly_stats,
    parse_params,
    project,
    project_model_sizes,
    get_model_at_quarter,
    _default_model_stats,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'notable_ai_models.csv')


def test_load_notable_models() -> None:
    df = load_notable_models(DATA_PATH)
    assert len(df) > 0
    assert "Parameters" in df.columns or "parameters" in df.columns.str.lower()


def test_compute_yearly_stats() -> None:
    df = load_notable_models(DATA_PATH)
    yearly = compute_yearly_stats(df)
    assert len(yearly) > 0


def test_project_model_sizes_length() -> None:
    cfg = load_config(CONFIG_PATH)
    model_sizes = project_model_sizes(cfg)

    expected_len = cfg.simulation.num_quarters
    assert len(model_sizes) == expected_len


def test_model_sizes_positive() -> None:
    cfg = load_config(CONFIG_PATH)
    model_sizes = project_model_sizes(cfg)

    for s in model_sizes:
        assert s > 0, "Model sizes must be positive"


def test_model_sizes_generally_increase() -> None:
    cfg = load_config(CONFIG_PATH)
    model_sizes = project_model_sizes(cfg)

    # Compare first year average to last year average
    qpy = cfg.simulation.quarters_per_year
    first_year = model_sizes[:qpy]
    last_year = model_sizes[-qpy:]
    assert sum(last_year) / len(last_year) >= sum(first_year) / len(first_year)


def test_get_model_at_quarter_returns_valid() -> None:
    cfg = load_config(CONFIG_PATH)
    model_sizes = project_model_sizes(cfg)

    for q in [0, 10, 20, 30, 40, 50]:
        if q < len(model_sizes):
            info = get_model_at_quarter(q, model_sizes, cfg)
            assert info.primary_model_size > 0
            assert info.distilled_model_size > 0
            assert 0 <= info.migration_fraction <= 1


def test_distilled_smaller_than_primary() -> None:
    cfg = load_config(CONFIG_PATH)
    model_sizes = project_model_sizes(cfg)

    # After initial quarters, distilled should be smaller
    for q in [10, 20, 30]:
        if q < len(model_sizes):
            info = get_model_at_quarter(q, model_sizes, cfg)
            assert info.distilled_model_size <= info.primary_model_size


# ---------------------------------------------------------------------------
# parse_params tests
# ---------------------------------------------------------------------------

class TestParseParamsSuffixes:
    """Standard suffix notation: number + B/M/K/G/T."""

    def test_billions(self) -> None:
        assert parse_params("70B") == 70e9

    def test_billions_decimal(self) -> None:
        assert parse_params("1.5B") == 1.5e9

    def test_millions(self) -> None:
        assert parse_params("350M") == 350e6

    def test_thousands(self) -> None:
        assert parse_params("500K") == 500e3

    def test_giga_alias(self) -> None:
        """'G' is an alias for billions (1e9)."""
        assert parse_params("7G") == 7e9

    def test_trillions(self) -> None:
        assert parse_params("1.2T") == 1.2e12

    def test_no_suffix(self) -> None:
        """Bare number string with no suffix."""
        assert parse_params("12345") == 12345.0


class TestParseParamsCaseInsensitive:
    """Suffixes should work in any case."""

    def test_lowercase_b(self) -> None:
        assert parse_params("70b") == 70e9

    def test_lowercase_m(self) -> None:
        assert parse_params("350m") == 350e6

    def test_lowercase_k(self) -> None:
        assert parse_params("500k") == 500e3

    def test_lowercase_t(self) -> None:
        assert parse_params("1t") == 1e12


class TestParseParamsWhitespaceAndCommas:
    """Whitespace and comma handling."""

    def test_leading_trailing_whitespace(self) -> None:
        assert parse_params("  70B  ") == 70e9

    def test_space_before_suffix(self) -> None:
        assert parse_params("70 B") == 70e9

    def test_commas_stripped(self) -> None:
        assert parse_params("1,000,000") == 1e6

    def test_commas_with_suffix(self) -> None:
        assert parse_params("1,000M") == 1e9


class TestParseParamsNumericInput:
    """Numeric (int/float) pass-through."""

    def test_int(self) -> None:
        assert parse_params(70) == 70.0

    def test_float(self) -> None:
        assert parse_params(1.5e9) == 1.5e9

    def test_zero(self) -> None:
        assert parse_params(0) == 0.0

    def test_negative(self) -> None:
        assert parse_params(-5) == -5.0


class TestParseParamsNoneAndInvalid:
    """Inputs that should return None."""

    def test_none(self) -> None:
        assert parse_params(None) is None

    def test_nan_float(self) -> None:
        assert parse_params(float('nan')) is None

    def test_empty_string(self) -> None:
        assert parse_params("") is None

    def test_garbage(self) -> None:
        assert parse_params("hello") is None

    def test_bool(self) -> None:
        # bool is subclass of int in Python, but should not be treated
        # as a parameter count
        assert parse_params(True) is None
        assert parse_params(False) is None

    def test_list(self) -> None:
        assert parse_params([1, 2, 3]) is None

    def test_dict(self) -> None:
        assert parse_params({"a": 1}) is None


class TestParseParamsEdgeCases:
    """Edge cases and unusual but valid inputs."""

    def test_decimal_no_integer_part(self) -> None:
        assert parse_params(".5B") == 0.5e9

    def test_large_number_string(self) -> None:
        assert parse_params("999999999999") == 999999999999.0

    def test_numpy_int(self) -> None:
        assert parse_params(np.int64(42)) == 42.0

    def test_numpy_float(self) -> None:
        assert parse_params(np.float64(1.5e9)) == 1.5e9

    def test_numpy_nan(self) -> None:
        assert parse_params(np.float64('nan')) is None


# ---------------------------------------------------------------------------
# load_notable_models — error paths and explicit column names
# ---------------------------------------------------------------------------

class TestLoadNotableModelsErrors:

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_notable_models("/nonexistent/path.csv")

    def test_bad_param_column(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError, match="Cannot identify parameter column"):
            load_notable_models(str(csv))

    def test_bad_date_column(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("Parameters,foo\n70B,x\n")
        with pytest.raises(ValueError, match="Cannot identify date column"):
            load_notable_models(str(csv))

    def test_explicit_column_names(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("p,d,n\n70B,2023-01-01,GPT\n1.5B,2022-06-15,BERT\n")
        df = load_notable_models(
            str(csv), col_params="p", col_date="d", col_name="n"
        )
        assert len(df) == 2
        assert set(df.columns) == {"year", "model_name", "parameters"}
        assert "GPT" in df["model_name"].values

    def test_numeric_date_column(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("Parameters,year\n70B,2023\n1B,2022\n")
        df = load_notable_models(str(csv))
        assert set(df["year"].unique()) == {2022, 2023}

    def test_no_name_column_defaults_to_unknown(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("Parameters,year\n70B,2023\n")
        df = load_notable_models(str(csv))
        assert df["model_name"].iloc[0] == "unknown"

    def test_min_year_filters(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("Parameters,year\n70B,2010\n1B,2020\n")
        df = load_notable_models(str(csv), min_year=2015)
        assert len(df) == 1
        assert df["year"].iloc[0] == 2020

    def test_unparseable_params_dropped(self, tmp_path) -> None:
        csv = tmp_path / "models.csv"
        csv.write_text("Parameters,year\n70B,2023\nhello,2023\n,2023\n")
        df = load_notable_models(str(csv))
        assert len(df) == 1


# ---------------------------------------------------------------------------
# compute_yearly_stats
# ---------------------------------------------------------------------------

class TestComputeYearlyStats:

    def test_columns(self) -> None:
        df = pd.DataFrame({
            "year": [2023, 2023, 2024],
            "model_name": ["a", "b", "c"],
            "parameters": [1e9, 3e9, 5e9],
        })
        stats = compute_yearly_stats(df)
        assert list(stats.columns) == ["p50", "p99", "mean"]
        assert 2023 in stats.index
        assert 2024 in stats.index

    def test_single_model_year(self) -> None:
        df = pd.DataFrame({
            "year": [2023],
            "model_name": ["x"],
            "parameters": [7e9],
        })
        stats = compute_yearly_stats(df)
        assert stats.loc[2023, "p50"] == pytest.approx(7e9)
        assert stats.loc[2023, "mean"] == pytest.approx(7e9)


# ---------------------------------------------------------------------------
# project — all three modes and error
# ---------------------------------------------------------------------------

class TestProject:

    def setup_method(self) -> None:
        self.known_years = np.array([2020.0, 2021.0, 2022.0])
        self.values = np.array([1e9, 2e9, 3e9])
        self.future_years = np.array([2023.0, 2024.0])

    def test_flat(self) -> None:
        out = project(self.values, self.known_years,
                      self.future_years, ProjectionMode.FLAT)
        assert len(out) == 5
        assert out[3] == pytest.approx(3e9)
        assert out[4] == pytest.approx(3e9)

    def test_linear(self) -> None:
        out = project(self.values, self.known_years,
                      self.future_years, ProjectionMode.LINEAR)
        assert len(out) == 5
        # Linear extrapolation of 1e9/yr slope
        assert out[3] > self.values[-1]

    def test_exponential(self) -> None:
        vals = np.array([1e9, 2e9, 4e9])  # doubling each year
        out = project(vals, self.known_years,
                      self.future_years, ProjectionMode.EXPONENTIAL)
        assert len(out) == 5
        assert out[3] > vals[-1]

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown projection mode"):
            project(self.values, self.known_years,
                    self.future_years, "bogus")


# ---------------------------------------------------------------------------
# project_model_sizes — fallback + contraction
# ---------------------------------------------------------------------------

class TestProjectModelSizes:

    def test_fallback_when_csv_missing(self) -> None:
        """Bad CSV path triggers fallback to _default_model_stats."""
        cfg = load_config(CONFIG_PATH)
        cfg.model_trends.notable_models_csv = "/nonexistent.csv"
        sizes = project_model_sizes(cfg)
        assert len(sizes) == cfg.simulation.num_quarters
        assert all(s > 0 for s in sizes)

    def test_contraction_caps_model_sizes(self) -> None:
        cfg = load_config(CONFIG_PATH)
        cfg.scenarios.model_size_contraction.enabled = True
        cfg.scenarios.model_size_contraction.contraction_year = 3
        cfg.scenarios.model_size_contraction.beta = 0.5
        sizes = project_model_sizes(cfg)
        qpy = cfg.simulation.quarters_per_year
        contraction_q = 3 * qpy
        cap = sizes[contraction_q]
        # All post-contraction quarters should equal the cap
        for q in range(contraction_q, len(sizes)):
            assert sizes[q] == pytest.approx(cap)
        # Cap should be smaller than uncapped value
        cfg2 = load_config(CONFIG_PATH)
        cfg2.scenarios.model_size_contraction.enabled = False
        sizes_no_cap = project_model_sizes(cfg2)
        if contraction_q < len(sizes_no_cap):
            assert cap <= sizes_no_cap[contraction_q]

    def test_contraction_year_beyond_simulation(self) -> None:
        """Contraction year past sim end should have no effect."""
        cfg = load_config(CONFIG_PATH)
        cfg.scenarios.model_size_contraction.enabled = True
        cfg.scenarios.model_size_contraction.contraction_year = 999
        sizes_contracted = project_model_sizes(cfg)
        cfg2 = load_config(CONFIG_PATH)
        cfg2.scenarios.model_size_contraction.enabled = False
        sizes_baseline = project_model_sizes(cfg2)
        assert sizes_contracted == sizes_baseline


# ---------------------------------------------------------------------------
# _default_model_stats
# ---------------------------------------------------------------------------

class TestDefaultModelStats:

    def test_returns_dataframe(self) -> None:
        stats = _default_model_stats()
        assert isinstance(stats, pd.DataFrame)
        assert "p50" in stats.columns
        assert len(stats) > 0

    def test_years_range(self) -> None:
        stats = _default_model_stats()
        assert 2015 in stats.index
        assert 2025 in stats.index


# ---------------------------------------------------------------------------
# get_model_at_quarter — migration and distillation logic
# ---------------------------------------------------------------------------

class TestGetModelAtQuarter:

    def test_first_quarter_uses_migration_fraction(self) -> None:
        cfg = load_config(CONFIG_PATH)
        sizes = project_model_sizes(cfg)
        info = get_model_at_quarter(0, sizes, cfg)
        assert info.migration_fraction == cfg.workload.initial_migration_fraction

    def test_q0_and_q1_are_migration_quarters(self) -> None:
        cfg = load_config(CONFIG_PATH)
        sizes = project_model_sizes(cfg)
        for q in [0, 1]:
            info = get_model_at_quarter(q, sizes, cfg)
            assert info.migration_fraction == cfg.workload.initial_migration_fraction

    def test_q2_onward_fully_migrated(self) -> None:
        cfg = load_config(CONFIG_PATH)
        sizes = project_model_sizes(cfg)
        info = get_model_at_quarter(2, sizes, cfg)
        assert info.migration_fraction == 1.0

    def test_distilled_size_is_fraction_of_primary_after_q2(self) -> None:
        cfg = load_config(CONFIG_PATH)
        sizes = project_model_sizes(cfg)
        frac = cfg.model_trends.distilled_model_size_fraction
        info = get_model_at_quarter(2, sizes, cfg)
        # After q2, both primary and distilled are current_size * frac
        assert info.distilled_model_size == pytest.approx(info.primary_model_size)
        # And both should be the distilled fraction of the full model
        full_size = sizes[2]
        assert info.primary_model_size == pytest.approx(full_size * frac)

    def test_year_boundary_resets_migration(self) -> None:
        """First 2 quarters of each year should be migration quarters."""
        cfg = load_config(CONFIG_PATH)
        sizes = project_model_sizes(cfg)
        qpy = cfg.simulation.quarters_per_year
        # Year 2, quarter 0 → q = 2*4 = 8
        q = 2 * qpy
        info = get_model_at_quarter(q, sizes, cfg)
        assert info.migration_fraction == cfg.workload.initial_migration_fraction
        # Year 2, quarter 2 → q = 10
        info2 = get_model_at_quarter(q + 2, sizes, cfg)
        assert info2.migration_fraction == 1.0

    def test_float_string_no_suffix(self) -> None:
        assert parse_params("3.14") == pytest.approx(3.14)

    def test_scientific_notation_string(self) -> None:
        """Exponential notation like '1e9' — regex won't match suffix,
        falls through to float() conversion."""
        assert parse_params("1e9") == 1e9

    def test_suffix_only(self) -> None:
        """Just a suffix letter with no number."""
        assert parse_params("B") is None

    def test_multiple_decimals(self) -> None:
        """'1.2.3B' is not a valid number."""
        assert parse_params("1.2.3B") is None
