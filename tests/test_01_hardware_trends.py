"""Tests mimicking notebook 01_hardware_trends."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy

import matplotlib
matplotlib.use("Agg")

from dc_tco.config import (
    load_config,
    ProjectionMode,
)
from dc_tco.hardware import build_gpu_roadmap, build_server_roadmap
from dc_tco.plotting import plot_gpu_trends


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')


def test_gpu_roadmap_has_known_and_projected():
    cfg = load_config(CONFIG_PATH)
    gpu_roadmap = build_gpu_roadmap(cfg)

    assert len(gpu_roadmap) > 0
    known = [g for g in gpu_roadmap if not g.code_name.startswith("Future")]
    projected = [g for g in gpu_roadmap if g.code_name.startswith("Future")]
    assert len(known) >= 7, "Expected at least 7 known GPUs (P100-B200)"
    assert len(projected) > 0, "Expected projected future generations"


def test_gpu_specs_positive():
    cfg = load_config(CONFIG_PATH)
    gpu_roadmap = build_gpu_roadmap(cfg)

    for g in gpu_roadmap:
        assert g.tdp > 0
        assert g.tflops > 0
        assert g.mem_bw > 0
        assert g.mem_cap > 0
        assert g.cost > 0


def test_plot_gpu_trends_returns_figure():
    cfg = load_config(CONFIG_PATH)
    gpu_roadmap = build_gpu_roadmap(cfg)

    fig = plot_gpu_trends(gpu_roadmap)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_server_roadmap_matches_gpu_roadmap():
    cfg = load_config(CONFIG_PATH)
    gpu_roadmap = build_gpu_roadmap(cfg)
    server_roadmap = build_server_roadmap(cfg)

    assert len(server_roadmap) == len(gpu_roadmap)
    for s in server_roadmap:
        assert s.tdp > 0
        assert s.tflops > 0
        assert s.cost > 0


def test_server_aggregates_gpus():
    cfg = load_config(CONFIG_PATH)
    server_roadmap = build_server_roadmap(cfg)

    for s in server_roadmap:
        # tflops in config is already per-server, not per-GPU
        assert s.tflops == s.gpu.tflops


def test_exponential_projection_mode():
    cfg = load_config(CONFIG_PATH)
    cfg_exp = copy.deepcopy(cfg)
    cfg_exp.hardware.projection.tflops = ProjectionMode.EXPONENTIAL
    cfg_exp.hardware.projection.mem_bw = ProjectionMode.EXPONENTIAL

    gpu_exp = build_gpu_roadmap(cfg_exp)
    gpu_lin = build_gpu_roadmap(cfg)

    # Exponential projections should diverge upward for later gens
    last_exp = gpu_exp[-1]
    last_lin = gpu_lin[-1]
    assert last_exp.tflops >= last_lin.tflops
    assert last_exp.mem_bw >= last_lin.mem_bw


def test_plot_gpu_trends_metric_subset():
    cfg = load_config(CONFIG_PATH)
    gpu_roadmap = build_gpu_roadmap(cfg)

    fig = plot_gpu_trends(gpu_roadmap, metrics=["tflops", "mem_bw"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
