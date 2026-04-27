"""Tests for hardware module."""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from dc_tco.config import ProjectionMode, load_config
from dc_tco.hardware import (
    GpuSpec,
    build_gpu_roadmap,
    build_server_roadmap,
    filter_roadmap,
    gpu_to_server,
    project,
)


@pytest.fixture
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    return load_config(config_path)


def test_project_flat():
    values = np.array([10.0, 20.0, 30.0])
    known_years = np.array([0.0, 1.0, 2.0])
    future_years = np.array([3.0, 4.0])
    result = project(values, known_years, future_years, mode=ProjectionMode.FLAT)
    # Flat returns known + future; future values equal the last known value
    assert len(result) == 5
    assert result[3] == 30.0
    assert result[4] == 30.0


def test_project_linear():
    values = np.array([10.0, 20.0, 30.0])
    known_years = np.array([0.0, 1.0, 2.0])
    future_years = np.array([3.0, 4.0])
    result = project(values, known_years, future_years, mode=ProjectionMode.LINEAR)
    assert len(result) == 5
    assert abs(result[3] - 40.0) < 1.0
    assert abs(result[4] - 50.0) < 1.0


def test_project_exponential():
    values = np.array([100.0, 200.0, 400.0])
    known_years = np.array([0.0, 1.0, 2.0])
    future_years = np.array([3.0])
    result = project(values, known_years, future_years, mode=ProjectionMode.EXPONENTIAL)
    assert len(result) == 4
    assert result[3] > 500.0  # ~800


def test_build_gpu_roadmap(cfg):
    roadmap = build_gpu_roadmap(cfg)
    assert len(roadmap) >= 7  # At least the 7 known GPUs
    names = [g.code_name for g in roadmap]
    assert "P100" in names
    assert "B200" in names
    for i in range(1, len(roadmap)):
        assert roadmap[i].release_quarter >= roadmap[i-1].release_quarter


def test_gpu_to_server(cfg):
    gpu = GpuSpec(
        code_name="TestGPU",
        release_quarter=0,
        tdp=300.0,  # Watts per GPU
        tflops=100.0,
        mem_bw=1000.0,
        mem_cap=80.0,  # GB per GPU
        net_bw=300.0,  # GB/s NVLink per GPU
        cost=10000.0,  # USD per GPU
    )
    server = gpu_to_server(gpu, cfg)
    assert server.cost == 10_000.0 * cfg.hardware.gpus_per_server
    # tflops in config is already per-server, passed through as-is
    assert server.tflops == 100.0
    expected_tdp = 300.0 * cfg.hardware.gpus_per_server / cfg.hardware.power_overhead_factor
    assert abs(server.tdp - expected_tdp) < 0.01


def test_filter_roadmap(cfg):
    roadmap = build_server_roadmap(cfg)
    filtered = filter_roadmap(roadmap, skip_generations=["H100"])
    names = [s.code_name for s in filtered]
    assert "H100" not in names
    assert "P100" in names
    assert "B200" in names


def test_server_roadmap_matches_gpu_count(cfg):
    gpu_roadmap = build_gpu_roadmap(cfg)
    server_roadmap = build_server_roadmap(cfg)
    assert len(gpu_roadmap) == len(server_roadmap)
