"""Networking configuration and cost lookups."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class NetworkCost:
    """Per-server networking costs."""
    capex_per_server: float
    opex_per_server_per_year: float
    performance_multiplier: float


def get_network_cost(cfg: Config) -> NetworkCost:
    """Look up the active networking configuration.

    Returns
    -------
    NetworkCost
        Per-server CapEx, OpEx, and performance multiplier for the
        selected network type.
    """
    net_cfg = cfg.networking.active
    return NetworkCost(
        capex_per_server=net_cfg.capex_per_server,
        opex_per_server_per_year=net_cfg.opex_per_server_per_year,
        performance_multiplier=net_cfg.performance_multiplier,
    )
