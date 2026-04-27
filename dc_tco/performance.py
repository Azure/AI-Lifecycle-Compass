"""Roofline-based performance model for LLM inference throughput."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import Config, PerformanceModelType
from .hardware import ServerSpec

# Cache for loaded CSV data — use Any for DataFrame type hint to avoid
# requiring pandas at import time.
_csv_cache: Dict[str, Any] = {}


def _resolve_csv_path(csv_path: str, config_dir: Optional[Path] = None) -> Path:
    """Resolve CSV path relative to config directory or CWD."""
    p = Path(csv_path)
    if p.is_absolute():
        return p
    if config_dir is not None:
        candidate = config_dir / p
        if candidate.exists():
            return candidate
    return p


def _load_throughput_csv(
    csv_path: str,
    config_dir: Optional[Path] = None,
) -> Any:
    """Load and cache a throughput CSV.

    Expected columns: model_size_B, gpu_type, throughput_tps
    Optional columns: batch_size, precision, seq_len
    """
    import pandas as pd

    resolved = str(_resolve_csv_path(csv_path, config_dir))
    if resolved in _csv_cache:
        return _csv_cache[resolved]

    df = pd.read_csv(resolved)
    if df.empty:
        raise ValueError(f"Throughput CSV is empty: {resolved}")
    required = {'model_size_B', 'gpu_type', 'throughput_tps'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Throughput CSV missing columns: {missing}. "
            f"Required: {required}"
        )
    _csv_cache[resolved] = df
    return df


def predict_throughput_csv(
    model_size_B: float,
    server: ServerSpec,
    cfg: Config,
    config_dir: Optional[Path] = None,
    oversubscription_factor: float = 1.0,
    network_multiplier: float = 1.0,
) -> float:
    """Predict throughput by interpolating from a CSV lookup table.

    Falls back to simplified model for unknown GPU types, preserving
    oversubscription_factor and network_multiplier for consistency.
    """
    csv_path = cfg.performance.throughput_csv
    if csv_path is None:
        return predict_throughput_simplified(
            model_size_B, server,
            oversubscription_factor, network_multiplier,
        )

    df = _load_throughput_csv(csv_path, config_dir)
    gpu_rows = df[df['gpu_type'] == server.code_name]

    if gpu_rows.empty:
        # Fallback: use simplified roofline for unknown GPU types
        return predict_throughput_simplified(
            model_size_B, server,
            oversubscription_factor, network_multiplier,
        )

    sizes = gpu_rows['model_size_B'].values
    tps = gpu_rows['throughput_tps'].values

    if len(sizes) == 1:
        tps0 = tps[0]
        return float(tps0)

    # Sort by model size for interpolation
    order = np.argsort(sizes)
    sizes = sizes[order]
    tps = tps[order]

    # Linear interpolation (clamp to range)
    result = float(np.interp(model_size_B, sizes, tps))
    return max(result, 1e-3)


# ---------------------------------------------------------------------------
# Analytical model: FLOPs and memory per token
# ---------------------------------------------------------------------------
def estimate_model_dims(model_size_B: float):
    """Rough scaling law for decoder-only transformers."""
    N = model_size_B * 1e9

    num_layers = int(0.8 * (model_size_B ** 0.5) * 100)  # ~80 for 70B
    hidden_dim = int((N / (12 * num_layers)) ** 0.5) * 12  # align to typical structure

    return num_layers, hidden_dim


def flops_per_token(model_size_B: float) -> float:
    """Estimate FLOPs required per token for a decoder-only transformer.

    Uses the standard approximation: ~2.1N FLOPs per token for a model
    with N parameters (accounts for forward pass compute) as linear layers (QKV, O, MLP) dominate.

    Parameters
    ----------
    model_size_B : float
        Model size in billions of parameters.
    """
    N = model_size_B * 1e9
    return 2.2 * N  # Slightly higher than 2.1 to account for additional overhead (softmax, layer norms, etc.)


def bytes_per_token(
    model_size_B: float,
    seq_len: int = 4000
) -> float:
    """Memory traffic per token (decode)."""
    N = model_size_B * 1e9
    bytes_per_elem = 2  # FP16

    # --- Weight streaming ---
    weight_bytes = N * bytes_per_elem

    # --- KV cache ---
    num_layers, hidden_dim = estimate_model_dims(model_size_B)

    kv_read = 2 * seq_len * hidden_dim * num_layers * bytes_per_elem
    kv_write = 2 * hidden_dim * num_layers * bytes_per_elem

    return float(weight_bytes + kv_read + kv_write)


def interconnect_bytes_per_token(model_size_B: float) -> float:
    """Interconnect traffic per token for tensor-parallel inference.

    Each transformer layer requires 2 all-reduce ops (attention + MLP),
    each transferring 2 * hidden_dim * bytes_per_elem bytes.
    """
    bytes_per_elem = 2  # FP16
    num_all_reduces = 2
    num_layers, hidden_dim = estimate_model_dims(model_size_B)
    # 2 all-reduces per layer, each sends 2 * hidden_dim elements
    return float(2 * num_all_reduces * hidden_dim * num_layers * bytes_per_elem)

# ---------------------------------------------------------------------------
# Throughput prediction
# ---------------------------------------------------------------------------


def predict_throughput_theoretical(
    model_size_B: float,
    server: ServerSpec,
    cfg: Config,
    seq_len: int = 4000
) -> float:
    """Predict steady-state throughput (tokens/s) using a multi-resource roofline model.

    Considers compute, memory bandwidth, and interconnect bandwidth limits.
    Enforces per-token latency SLOs (TBT and TTFT).
    """

    perf = cfg.performance

    # --- Per-token costs ---
    fpt = flops_per_token(model_size_B)  # FLOPs/token
    bpt = bytes_per_token(model_size_B, seq_len)  # bytes/token (KV + activations)
    npt = interconnect_bytes_per_token(model_size_B)  # bytes/token (all-reduce)

    # --- Compute bound ---
    compute_tps = (
        perf.compute_efficiency * (server.tflops * 1e12) / fpt
    )

    # --- Memory bound ---
    mem_tps = (
        perf.memory_efficiency * (server.mem_bw * 1e9) / bpt
    )

    # --- Interconnect bound ---
    net_tps = float("inf")
    if server.net_bw > 0 and npt > 0:
        net_tps = perf.network_efficiency * (server.net_bw * 1e9) / npt

    # --- Roofline: bottleneck resource ---
    token_rate = min(compute_tps, mem_tps, net_tps)

    # Enforce SLO: if per-token latency exceeds TBT or TTFT, cap throughput
    if token_rate > 0:
        per_token_latency = 1.0 / token_rate
        if per_token_latency > perf.max_tbt_s:
            token_rate = 1.0 / perf.max_tbt_s
        elif per_token_latency > perf.max_ttft_s:
            token_rate = 1.0 / perf.max_ttft_s

    return max(token_rate, 1e-3)


def predict_throughput_simplified(
    model_size_B: float,
    server: ServerSpec,
    oversubscription_factor: float = 1.0,
    network_multiplier: float = 1.0,
) -> float:
    """Simplified per-server goodput used by the lifecycle simulator.

    This simplified model accounts for the primary matrix multiplications (MatMuls) in the transformer blocks
    which equals 2 FLOPs per parameter:
        goodput = min(mem_bw, TFLOPS * 1000) / (2 * model_size_B)

    Parameters
    ----------
    model_size_B : float
        Model size in billions of parameters.
    server : ServerSpec
        Server specification (already at server level).
    oversubscription_factor : float
        Oversubscription factor (default 1.0, no oversubscription).
    network_multiplier : float
        Network performance multiplier (1.0 for NVLink, 0.3 for Ethernet).
    """
    MATMUL_OPS = 2.0  # 2 floating operations per parameter for the main MatMul operations
    server_gflops = server.tflops * 1000
    num_ops = MATMUL_OPS * model_size_B  # FLOPs per token
    perf_mem = server.mem_bw / num_ops  # GB/s / num_ops -> tokens/s
    perf_compute = server_gflops / num_ops  # GFLOPS / num_ops -> tokens/s

    # Interconnect bound: all-reduce bytes per token across GPUs
    npt = interconnect_bytes_per_token(model_size_B)
    perf_net = float("inf")
    if server.net_bw > 0 and npt > 0:
        perf_net = (server.net_bw * network_multiplier) / (npt / 1e9)  # GB/s / GB -> tokens/s

    goodput = min(perf_mem, perf_compute, perf_net) * oversubscription_factor
    return goodput


def predict_throughput(
    model_size_B: float,
    server: ServerSpec,
    cfg: Config,
    network_multiplier: float = 1.0,
) -> float:
    """Predict per-server throughput using the configured model.

    Parameters
    ----------
    model_size_B : float
        Model size in billions of parameters.
    server : ServerSpec
        Server specification.
    cfg : Config
        Configuration.
    network_multiplier : float
        Network performance multiplier.

    Returns
    -------
    float
        Tokens per second (per server).
    """
    if cfg.performance.model_type == PerformanceModelType.THEORETICAL:
        return predict_throughput_theoretical(model_size_B, server, cfg)
    if cfg.performance.model_type == PerformanceModelType.CSV:
        return predict_throughput_csv(
            model_size_B, server, cfg,
            oversubscription_factor=cfg.policy.oversubscription_factor,
            network_multiplier=network_multiplier,
        )
    # Simplified model (used by lifecycle simulator)
    return predict_throughput_simplified(
        model_size_B, server,
        cfg.policy.oversubscription_factor,
        network_multiplier,
    )
