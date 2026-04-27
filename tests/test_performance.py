"""Tests for performance module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from dc_tco.hardware import GpuSpec, gpu_to_server
from dc_tco.config import load_config
from dc_tco.performance import (
    flops_per_token,
    bytes_per_token,
    predict_throughput_simplified,
    predict_throughput_theoretical,
)


@pytest.fixture
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    return load_config(config_path)


@pytest.fixture
def test_server(cfg):
    gpu = GpuSpec(
        code_name="TestH100",
        release_quarter=0,
        tdp=700.0,  # Watts per GPU
        tflops=267.6,  # FP64 per server
        mem_bw=3440.0,  # GB/s per GPU
        mem_cap=80.0,  # GB per GPU
        net_bw=300.0,  # GB/s NVLink per GPU
        cost=27000.0,  # USD per GPU
    )
    return gpu_to_server(gpu, cfg)


def test_flops_per_token():
    # 70B model: 2.2 * 70e9 = 154e9 FLOPs
    result = flops_per_token(70.0)
    assert abs(result - 154e9) < 1e6


def test_bytes_per_token():
    result = bytes_per_token(70.0)
    assert result > 0


def test_simplified_throughput_positive(test_server):
    tp = predict_throughput_simplified(
        model_size_B=70.0,
        server=test_server,
        oversubscription_factor=1.5,
        network_multiplier=1.0,
    )
    assert tp > 0


def test_simplified_throughput_scales_with_model_size(test_server):
    # Larger model should have lower throughput
    tp_small = predict_throughput_simplified(10.0, test_server, 1.0, 1.0)
    tp_large = predict_throughput_simplified(100.0, test_server, 1.0, 1.0)
    assert tp_small > tp_large


def test_simplified_throughput_scales_with_oversubscription(test_server):
    tp_1x = predict_throughput_simplified(70.0, test_server, 1.0, 1.0)
    tp_2x = predict_throughput_simplified(70.0, test_server, 2.0, 1.0)
    assert abs(tp_2x / tp_1x - 2.0) < 0.01


def test_theoretical_throughput_positive(test_server, cfg):
    tp = predict_throughput_theoretical(
        model_size_B=70.0,
        server=test_server,
        cfg=cfg,
    )
    assert tp > 0


# ---------------------------------------------------------------------------
# Tests that verify GPU specs from default.yaml and derived server/throughput
# ---------------------------------------------------------------------------

def test_h100_gpu_spec_from_config(cfg):
    """Verify H100 GPU spec values match default.yaml."""
    from dc_tco.hardware import build_gpu_roadmap
    roadmap = build_gpu_roadmap(cfg)
    h100 = next(g for g in roadmap if g.code_name == "H100")

    assert h100.tdp == 700  # Watts
    assert h100.tflops == 267.6  # FP64 TFLOPS per server
    assert h100.mem_bw == 3440  # GB/s per GPU
    assert h100.mem_cap == 80  # GB per GPU
    assert h100.cost == 27000  # USD per GPU


def test_h100_server_spec_from_config(cfg):
    """Verify H100 server-level aggregation (8 GPUs, overhead 0.5)."""
    from dc_tco.hardware import build_gpu_roadmap
    roadmap = build_gpu_roadmap(cfg)
    h100_gpu = next(g for g in roadmap if g.code_name == "H100")
    h100_srv = gpu_to_server(h100_gpu, cfg)

    n = cfg.hardware.gpus_per_server  # 8
    overhead = cfg.hardware.power_overhead_factor  # 0.5

    # tflops in config is already per-server, passed through as-is
    assert h100_srv.tflops == h100_gpu.tflops  # 267.6
    assert h100_srv.mem_bw == n * h100_gpu.mem_bw  # 27520
    assert h100_srv.mem_cap == n * h100_gpu.mem_cap  # 640
    assert h100_srv.cost == n * h100_gpu.cost  # 216000
    assert h100_srv.tdp == n * h100_gpu.tdp / overhead  # 11200
    assert h100_srv.tflops == 267.6  # TFLOPS per server (already per-server in config)
    assert h100_srv.mem_bw == 27520  # GB/s
    assert h100_srv.mem_cap == 640  # GB
    assert h100_srv.cost == 216000  # USD
    assert h100_srv.tdp == 11200  # Watts


def test_h100_simplified_throughput_70b(cfg):
    """Verify tokens/s for 70B model on H100 server (simplified model)."""
    from dc_tco.hardware import build_gpu_roadmap
    roadmap = build_gpu_roadmap(cfg)
    h100_gpu = next(g for g in roadmap if g.code_name == "H100")
    h100_srv = gpu_to_server(h100_gpu, cfg)

    tp = predict_throughput_simplified(70.0, h100_srv, 1.0, 1.0)

    # min(27520, 2140800) / (2 * 70) = 27520 / 140 = 196.57
    expected = min(h100_srv.mem_bw, h100_srv.tflops * 1000) / (2.0 * 70.0)
    assert abs(tp - expected) < 0.01
    assert tp > 100  # sanity: should be well above 100 tok/s


def test_a100_simplified_throughput_70b(cfg):
    """Verify tokens/s for 70B model on A100 server (simplified model)."""
    from dc_tco.hardware import build_gpu_roadmap
    roadmap = build_gpu_roadmap(cfg)
    a100_gpu = next(g for g in roadmap if g.code_name == "A100")
    a100_srv = gpu_to_server(a100_gpu, cfg)

    tp = predict_throughput_simplified(70.0, a100_srv, 1.0, 1.0)

    expected = min(a100_srv.mem_bw, a100_srv.tflops * 1000) / (2.0 * 70.0)
    assert abs(tp - expected) < 0.01


def test_newer_gpu_faster_throughput(cfg):
    """H100 server should have higher throughput than A100 for same model."""
    from dc_tco.hardware import build_gpu_roadmap
    roadmap = build_gpu_roadmap(cfg)

    a100_srv = gpu_to_server(next(g for g in roadmap if g.code_name == "A100"), cfg)
    h100_srv = gpu_to_server(next(g for g in roadmap if g.code_name == "H100"), cfg)

    tp_a100 = predict_throughput_simplified(70.0, a100_srv, 1.0, 1.0)
    tp_h100 = predict_throughput_simplified(70.0, h100_srv, 1.0, 1.0)

    assert tp_h100 > tp_a100
