"""Tests for TCO module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from dc_tco.config import load_config
from dc_tco.hardware import GpuSpec, gpu_to_server
from dc_tco.tco import (
    TcoBreakdown,
    ServerBatch,
    compute_racks,
    compute_batch_capex,
    compute_batch_opex,
)


@pytest.fixture
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    return load_config(config_path)


@pytest.fixture
def test_server(cfg):
    gpu = GpuSpec(
        code_name="TestGPU",
        release_quarter=0,
        tdp=700.0,
        tflops=267.6,
        mem_bw=3440.0,
        mem_cap=80.0,
        net_bw=300.0,
        cost=27000.0,
    )
    return gpu_to_server(gpu, cfg)


def test_tco_breakdown_totals():
    b = TcoBreakdown(
        capex_server=100.0,
        capex_rack=20.0,
        capex_power_provisioning=30.0,
        capex_cooling_provisioning=10.0,
        capex_network=5.0,
        opex_energy=50.0,
        opex_cooling=15.0,
        opex_maintenance=10.0,
        opex_network=5.0,
    )
    assert b.total_capex == 165.0
    assert b.total_opex == 80.0
    assert b.total == 245.0


def test_tco_breakdown_add():
    b1 = TcoBreakdown(capex_server=100.0, opex_energy=50.0)
    b2 = TcoBreakdown(capex_server=200.0, opex_energy=75.0)
    b3 = b1 + b2
    assert b3.capex_server == 300.0
    assert b3.opex_energy == 125.0


def test_compute_racks():
    # 10 servers at 11200W TDP, 50kW rack -> 4.48 servers/rack -> 3 racks
    n_racks = compute_racks(10, server_tdp=11200.0, rack_power_w=50000.0)
    assert n_racks >= 3


def test_compute_batch_capex(test_server, cfg):
    batch = ServerBatch(
        batch_id=0,
        server=test_server,
        start_q=0,
        decom_q=20,
        num_servers=10,
        num_racks=3,
    )
    capex = compute_batch_capex(batch, q=0, cfg=cfg)
    # CapEx should be positive only at the purchase quarter (amortized)
    assert capex.capex_server > 0

    # CapEx after purchase quarter should still be positive (amortized)
    capex_later = compute_batch_capex(batch, q=5, cfg=cfg)
    assert capex_later.capex_server > 0


def test_compute_batch_opex(test_server, cfg):
    batch = ServerBatch(
        batch_id=0,
        server=test_server,
        start_q=0,
        decom_q=20,
        num_servers=10,
        num_racks=3,
    )
    opex = compute_batch_opex(batch, q=5, cfg=cfg)
    assert opex.opex_energy > 0
    assert opex.opex_maintenance > 0

    # OpEx after decommission should be zero
    opex_after = compute_batch_opex(batch, q=25, cfg=cfg)
    assert opex_after.opex_energy == 0.0


def test_server_batch_active_at():
    batch = ServerBatch(
        batch_id=0,
        server=None,  # Not needed for active_at check
        start_q=5,
        decom_q=25,
        num_servers=10,
        num_racks=2,
    )
    assert not batch.active_at(4)
    assert batch.active_at(5)
    assert batch.active_at(24)
    assert not batch.active_at(25)
