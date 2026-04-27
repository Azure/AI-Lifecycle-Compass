"""GPU hardware database, trend projection, and server-level scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Union, Dict

import numpy as np

from .config import ProjectionMode

from .config import Config, GpuEntry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GpuSpec:
    """Single-GPU specification."""
    code_name: str
    release_quarter: int    # Quarter index relative to base_year (quarter 0)
    tdp: float              # Watts
    tflops: float           # FP16 TFLOPS <- this is wrong
    mem_bw: float           # GB/s
    mem_cap: float          # GB
    net_bw: float           # NVLink bisection bandwidth (GB/s per GPU)
    cost: float             # USD

    @property
    def release_year(self) -> float:
        return self.release_quarter / 4.0


@dataclass
class ServerSpec:
    """Server-level specification (multiple GPUs + overhead)."""
    gpu: GpuSpec
    gpus_per_server: int
    tdp: float              # Total server TDP (watts)
    tflops: float           # Total server TFLOPS
    mem_bw: float           # Total server mem bandwidth (GB/s)
    mem_cap: float          # Total server mem capacity (GB)
    net_bw: float           # Total server NVLink bisection bandwidth (GB/s)
    cost: float             # Total server cost (USD)

    @property
    def code_name(self) -> str:
        return self.gpu.code_name

    @property
    def release_quarter(self) -> int:
        return self.gpu.release_quarter


# ---------------------------------------------------------------------------
# Quarter-string parser
# ---------------------------------------------------------------------------

# Default used when no config is available (matches the bundled GPU dataset)
_DEFAULT_BASE_YEAR = 2015


def parse_quarter_string(qstr: str) -> float:
    """Convert quarter string like "Q2'21" to a fractional year (e.g., 21.25).

    Returns the *two-digit* year plus quarter offset:
      Q1 -> +0.0, Q2 -> +0.25, Q3 -> +0.5, Q4 -> +0.75
    """
    q_num = int(qstr[1])
    year_2d = int(qstr.split("'")[1])
    return year_2d + (q_num - 1) * 0.25


def year_to_quarter_index(year_frac: float, base_year: int = _DEFAULT_BASE_YEAR) -> int:
    """Convert fractional (two-digit) year to quarter index relative to *base_year*.

    *base_year* is a full calendar year (e.g. 2016); converted to two-digit
    form internally so the arithmetic matches ``parse_quarter_string`` output.
    """
    base_2d = base_year % 100
    return int(round((year_frac - base_2d) * 4))


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project(
    values: np.ndarray,
    known_years: np.ndarray,
    future_years: np.ndarray,
    mode: Union[str, ProjectionMode] = ProjectionMode.LINEAR,
) -> np.ndarray:
    """Extrapolate *GPU hardware values* from *known_years* into *future_years*.

    Parameters
    ----------
    mode : str or ProjectionMode
        ``"flat"`` — last known value repeated.
        ``"linear"`` — first-degree polynomial fit.
        ``"exponential"`` — linear fit in log-space, then exponentiated.
    """
    values = np.asarray(values, dtype=float)
    known_years = np.asarray(known_years, dtype=float)
    future_years = np.asarray(future_years, dtype=float)

    n_known = len(values)
    projected = np.zeros(n_known + len(future_years))
    projected[:n_known] = values

    if mode == ProjectionMode.FLAT:
        projected[n_known:] = values[-1]
    elif mode == ProjectionMode.LINEAR:
        coeffs = np.polyfit(known_years, values, 1)
        projected[n_known:] = np.polyval(coeffs, future_years)
        # Ensure no negative values for physical quantities
        projected[n_known:] = np.maximum(projected[n_known:], values[-1] * 0.5)
    elif mode == ProjectionMode.EXPONENTIAL:
        log_vals = np.log(np.maximum(values, 1e-10))
        slope, intercept = np.polyfit(known_years, log_vals, 1)
        projected[n_known:] = np.exp(intercept + slope * future_years)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")

    return projected


# ---------------------------------------------------------------------------
# Roadmap builders
# ---------------------------------------------------------------------------

def _gpu_entries_to_specs(entries: List[GpuEntry], base_year: int = _DEFAULT_BASE_YEAR) -> List[GpuSpec]:
    """Convert config GpuEntry list to GpuSpec list with computed quarter indices."""
    specs = []
    for e in entries:
        year_frac = parse_quarter_string(e.quarter)
        q_idx = year_to_quarter_index(year_frac, base_year)
        specs.append(GpuSpec(
            code_name=e.code_name,
            release_quarter=q_idx,
            tdp=e.tdp,
            tflops=e.tflops,
            mem_bw=e.mem_bw,
            mem_cap=e.mem_cap,
            net_bw=e.net_bw,
            cost=e.cost,
        ))
    return specs


def build_gpu_roadmap(cfg: Config) -> List[GpuSpec]:
    """Build full GPU roadmap: known GPUs + projected future generations.

    Returns a list of GpuSpec sorted by release_quarter.
    """
    base_year = cfg.simulation.base_year
    base_year_2d = base_year % 100

    known = _gpu_entries_to_specs(cfg.hardware.known_gpus, base_year)
    if not known:
        return []

    # Years of known GPUs (fractional, two-digit)
    known_years = np.array([parse_quarter_string(e.quarter) for e in cfg.hardware.known_gpus])

    # How many future GPUs to generate to cover the simulation horizon
    last_known_year = known_years[-1]
    sim_end_year = base_year_2d + cfg.simulation.total_years
    interval = cfg.hardware.release_interval_years
    assert interval > 0
    num_future = max(0, int(math.ceil((sim_end_year - last_known_year) / interval)))

    if num_future == 0:
        return sorted(known, key=lambda g: g.release_quarter)

    future_years = np.array([
        last_known_year + (i + 1) * interval
        for i in range(num_future)
    ])

    # Project each GPU metric
    proj_cfg = cfg.hardware.projection
    metrics = {
        'tdp': (np.array([g.tdp for g in known]), proj_cfg.tdp),
        'tflops': (np.array([g.tflops for g in known]), proj_cfg.tflops),
        'mem_bw': (np.array([g.mem_bw for g in known]), proj_cfg.mem_bw),
        'mem_cap': (np.array([g.mem_cap for g in known]), proj_cfg.mem_cap),
        'net_bw': (np.array([g.net_bw for g in known]), proj_cfg.net_bw),
        'cost': (np.array([g.cost for g in known]), proj_cfg.cost),
    }

    projected: Dict[str, np.ndarray] = {}
    for metric, (vals, mode) in metrics.items():
        full = project(vals, known_years, future_years, mode)
        projected[metric] = full[len(vals):]  # Only the future portion

    # Create future GpuSpec entries
    future_gpus: List[GpuSpec] = []
    for i in range(num_future):
        year_frac = future_years[i]
        q_idx = year_to_quarter_index(year_frac, base_year)
        future_gpus.append(GpuSpec(
            code_name=f"FutureGPU_{int(2000 + year_frac):.0f}",
            release_quarter=q_idx,
            tdp=max(projected['tdp'][i], 100),      # Floor to reasonable minimums
            tflops=max(projected['tflops'][i], 1),
            mem_bw=max(projected['mem_bw'][i], 100),
            mem_cap=max(projected['mem_cap'][i], 8),
            net_bw=max(projected['net_bw'][i], 100),
            cost=max(projected['cost'][i], 1000),
        ))

    all_gpus = known + future_gpus
    return sorted(all_gpus, key=lambda g: g.release_quarter)


def gpu_to_server(gpu: GpuSpec, cfg: Config) -> ServerSpec:
    """Scale a single-GPU spec to server-level, accounting for multi-GPU and power overhead."""
    n = cfg.hardware.gpus_per_server
    overhead = cfg.hardware.power_overhead_factor
    return ServerSpec(
        gpu=gpu,
        gpus_per_server=n,
        tdp=n * gpu.tdp / overhead,        # e.g., 8 * 700 / 0.5 = 11200W
        tflops=gpu.tflops,  # Keep in mind this is per server
        mem_bw=n * gpu.mem_bw,
        mem_cap=n * gpu.mem_cap,
        net_bw=n * gpu.net_bw,             # Total NVLink bandwidth across all GPUs
        cost=n * gpu.cost,                 # e.g., 8 * 27000 = $216,000
    )


def build_server_roadmap(cfg: Config) -> List[ServerSpec]:
    """Build full server roadmap from GPU roadmap."""
    gpu_roadmap = build_gpu_roadmap(cfg)
    return [
        gpu_to_server(g, cfg)
        for g in gpu_roadmap
    ]


def filter_roadmap(
    roadmap: List[ServerSpec],
    skip_generations: Optional[List[str]] = None,
) -> List[ServerSpec]:
    """Remove specified GPU generations from the roadmap."""
    if not skip_generations:
        return roadmap
    return [s for s in roadmap if s.code_name not in skip_generations]
