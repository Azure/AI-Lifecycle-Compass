"""Workload demand generation with scenario support."""

from __future__ import annotations

import numpy as np

from .config import Config, ScalingMode


def compute_demand(q: int, cfg: Config) -> float:
    """Compute demand (requests/second) at quarter *q*.

    Supports flat, linear, and exponential scaling, plus an optional
    demand shock (shock_plateau) scenario.

    Parameters
    ----------
    q : int
        Quarter index (0-based).
    cfg : Config
        Configuration with workload and scenario settings.

    Returns
    -------
    float
        Demand in requests per second.
    """
    wl = cfg.workload
    qpy = cfg.simulation.quarters_per_year

    # --- Base demand model ---
    # scaling_factor is a *per-quarter* rate (e.g. 1.05 = 5% per quarter).
    if wl.scaling == ScalingMode.FLAT:
        baseline = wl.initial_demand
    elif wl.scaling == ScalingMode.LINEAR:
        baseline = wl.initial_demand * (1.0 + (wl.scaling_factor - 1.0) * q)
    elif wl.scaling == ScalingMode.EXPONENTIAL:
        baseline = wl.initial_demand * (wl.scaling_factor ** q)
    else:
        raise ValueError(f"Unknown workload scaling mode: {wl.scaling}")

    # --- Demand shock scenario ---
    shock = cfg.scenarios.demand_shock
    if shock.enabled:
        shock_q = shock.shock_year * qpy
        if q >= shock_q:
            # Compute baseline at the shock quarter
            if wl.scaling == ScalingMode.LINEAR:
                shock_baseline = wl.initial_demand * (1.0 + (wl.scaling_factor - 1.0) * shock_q)
            elif wl.scaling == ScalingMode.EXPONENTIAL:
                shock_baseline = wl.initial_demand * (wl.scaling_factor ** shock_q)
            else:
                shock_baseline = wl.initial_demand

            shock_level = shock.alpha * shock_baseline

            # Post-shock growth (0 = plateau)
            if shock.post_shock_growth > 0:
                quarters_after = q - shock_q
                shock_level *= (1.0 + shock.post_shock_growth / qpy) ** quarters_after

            return shock_level

    return baseline


def build_demand_curve(cfg: Config) -> np.ndarray:
    """Build the full demand curve for all simulation quarters.

    Returns
    -------
    np.ndarray
        Array of demand values, one per quarter.
    """
    n = cfg.simulation.num_quarters
    return np.array([compute_demand(q, cfg) for q in range(n)])
