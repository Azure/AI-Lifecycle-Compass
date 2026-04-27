"""Tests for CSV-based throughput performance model."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dc_tco.config import Config
from dc_tco.hardware import GpuSpec, ServerSpec
from dc_tco.performance import (
    predict_throughput,
    predict_throughput_csv,
    _resolve_csv_path,
    _csv_cache,
)


@pytest.fixture(autouse=True)
def clear_csv_cache():
    """Clear CSV cache between tests."""
    _csv_cache.clear()
    yield
    _csv_cache.clear()


def _make_server(code_name: str = "H100") -> ServerSpec:
    gpu = GpuSpec(
        code_name=code_name,
        release_quarter=0,
        tdp=700,
        tflops=1979.0,
        mem_bw=3350.0,
        mem_cap=80.0,
        net_bw=300.0,
        cost=30000.0,
    )
    return ServerSpec(
        gpu=gpu,
        gpus_per_server=8,
        tdp=10200,
        tflops=15960.0,
        mem_bw=26400.0,
        mem_cap=640.0,
        net_bw=2400.0,
        cost=250000.0,
    )


def _write_csv(tmp_path: Path, content: str) -> Path:
    csv_file = tmp_path / "throughput.csv"
    csv_file.write_text(textwrap.dedent(content).strip())
    return csv_file


class TestResolveCsvPath:
    def test_absolute_path(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        resolved = _resolve_csv_path(str(p))
        assert resolved == p

    def test_relative_to_config_dir(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        resolved = _resolve_csv_path("data.csv", config_dir=tmp_path)
        assert resolved == tmp_path / "data.csv"

    def test_fallback_to_cwd(self):
        resolved = _resolve_csv_path("nonexistent.csv")
        assert resolved == Path("nonexistent.csv")


class TestPredictThroughputCsv:
    def test_exact_lookup(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
            175,H100,2000
        """)
        cfg = Config()
        cfg.performance.model_type = "csv"
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        result = predict_throughput_csv(70.0, server, cfg)
        assert result == 5000.0

    def test_interpolation(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
            175,H100,2000
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        # 122.5 is midpoint between 70 and 175 → should be ~3500
        result = predict_throughput_csv(122.5, server, cfg)
        assert abs(result - 3500.0) < 1.0

    def test_clamp_below(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
            175,H100,2000
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        # Below minimum → clamp to first value
        result = predict_throughput_csv(10.0, server, cfg)
        assert result == 5000.0

    def test_clamp_above(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
            175,H100,2000
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        result = predict_throughput_csv(500.0, server, cfg)
        assert result == 2000.0

    def test_unknown_gpu_falls_back(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("FutureGPU_3000")

        # Should fall back to simplified model
        result = predict_throughput_csv(70.0, server, cfg)
        assert result > 0

    def test_no_csv_path_falls_back(self):
        cfg = Config()
        cfg.performance.throughput_csv = None
        server = _make_server("H100")

        result = predict_throughput_csv(70.0, server, cfg)
        assert result > 0

    def test_single_data_point(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        result = predict_throughput_csv(70.0, server, cfg)
        assert result == 5000.0

    def test_missing_columns(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type
            70,H100
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        with pytest.raises(ValueError, match="missing columns"):
            predict_throughput_csv(70.0, server, cfg)


class TestPredictThroughputDispatcher:
    def test_csv_dispatch(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
        """)
        cfg = Config()
        cfg.performance.model_type = "csv"
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        result = predict_throughput(70.0, server, cfg)
        assert result == 5000.0

    def test_simplified_dispatch(self):
        cfg = Config()
        cfg.performance.model_type = "simplified"
        server = _make_server("H100")

        result = predict_throughput(70.0, server, cfg)
        assert result > 0


class TestCsvCaching:
    def test_cache_hit(self, tmp_path):
        csv = _write_csv(tmp_path, """
            model_size_B,gpu_type,throughput_tps
            70,H100,5000
        """)
        cfg = Config()
        cfg.performance.throughput_csv = str(csv)
        server = _make_server("H100")

        predict_throughput_csv(70.0, server, cfg)
        assert len(_csv_cache) == 1

        # Second call should use cache
        predict_throughput_csv(70.0, server, cfg)
        assert len(_csv_cache) == 1
