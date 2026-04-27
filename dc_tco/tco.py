"""TCO (Total Cost of Ownership) computation: CapEx and OpEx per server batch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .hardware import ServerSpec
from .networking import NetworkCost


@dataclass
class TcoBreakdown:
    """Itemized TCO for a single quarter."""
    # CapEx (amortized)
    capex_server: float = 0.0
    capex_rack: float = 0.0
    capex_power_provisioning: float = 0.0
    capex_cooling_provisioning: float = 0.0
    capex_facility: float = 0.0
    capex_network: float = 0.0

    # OpEx
    opex_energy: float = 0.0
    opex_cooling: float = 0.0
    opex_maintenance: float = 0.0
    opex_network: float = 0.0

    @property
    def total_capex(self) -> float:
        return (self.capex_server + self.capex_rack +
                self.capex_power_provisioning + self.capex_cooling_provisioning +
                self.capex_facility + self.capex_network)

    @property
    def total_opex(self) -> float:
        return (self.opex_energy + self.opex_cooling +
                self.opex_maintenance + self.opex_network)

    @property
    def total(self) -> float:
        return self.total_capex + self.total_opex

    def __iadd__(self, other: TcoBreakdown) -> TcoBreakdown:
        self.capex_server += other.capex_server
        self.capex_rack += other.capex_rack
        self.capex_power_provisioning += other.capex_power_provisioning
        self.capex_cooling_provisioning += other.capex_cooling_provisioning
        self.capex_facility += other.capex_facility
        self.capex_network += other.capex_network
        self.opex_energy += other.opex_energy
        self.opex_cooling += other.opex_cooling
        self.opex_maintenance += other.opex_maintenance
        self.opex_network += other.opex_network
        return self

    def __add__(self, other: TcoBreakdown) -> TcoBreakdown:
        result = TcoBreakdown()
        result += self
        result += other
        return result


# ---------------------------------------------------------------------------
# Per-batch TCO computation
# ---------------------------------------------------------------------------

@dataclass
class ServerBatch:
    """A batch of identical servers provisioned at the same time."""
    batch_id: int
    server: ServerSpec
    start_q: int             # Quarter provisioned
    decom_q: int             # Quarter decommissioned (exclusive)
    num_servers: int
    num_racks: int
    maintenance_multiplier: float = 1.0  # Increased for extended-lifetime servers

    @property
    def is_active(self) -> bool:
        return self.start_q < self.decom_q

    def active_at(self, q: int) -> bool:
        return self.start_q <= q < self.decom_q


def compute_racks(num_servers: int, server_tdp: float, rack_power_w: float) -> int:
    """Compute number of racks needed to house servers, given power constraints."""
    servers_per_rack = max(1, int(math.floor(rack_power_w / server_tdp)))
    return int(math.ceil(num_servers / servers_per_rack))


def compute_batch_capex(
    batch: ServerBatch,
    q: int,
    cfg: Config,
    net_cost: Optional[NetworkCost] = None,
) -> TcoBreakdown:
    """Compute CapEx for a server batch at quarter *q*.

    CapEx is amortized over the full amortization period, even if
    the server is decommissioned early.
    """
    tco_cfg = cfg.tco
    qpy = cfg.simulation.quarters_per_year
    amort_end_q = batch.start_q + tco_cfg.amortization_years * qpy

    result = TcoBreakdown()

    # Only charge CapEx during amortization window
    if not (batch.start_q <= q < amort_end_q):
        return result

    n = batch.num_servers
    amort = tco_cfg.amortization_years

    # Server CapEx: cost / amortization_years, per quarter
    result.capex_server = (batch.server.cost / amort) * n / qpy

    # Rack CapEx
    result.capex_rack = (tco_cfg.rack_cost / amort) * batch.num_racks / qpy

    # Network CapEx
    if net_cost is not None:
        result.capex_network = (net_cost.capex_per_server / amort) * n / qpy

    return result


def compute_batch_opex(
    batch: ServerBatch,
    q: int,
    cfg: Config,
    net_cost: Optional[NetworkCost] = None,
) -> TcoBreakdown:
    """Compute OpEx for a server batch at quarter *q*.

    OpEx is only charged while the server is active (before decommission).
    """
    if not batch.active_at(q):
        return TcoBreakdown()

    tco_cfg = cfg.tco
    qpy = cfg.simulation.quarters_per_year
    n = batch.num_servers
    result = TcoBreakdown()

    # Energy cost: server_power_kW * server_utilization * $/kWh * hours_per_quarter
    power_kw = (batch.server.tdp * n) / 1000.0
    hours_per_quarter = tco_cfg.hours_per_year / qpy
    result.opex_energy = power_kw * tco_cfg.utilization * tco_cfg.power_cost_per_kwh * hours_per_quarter

    # Cooling OpEx: fraction of energy cost
    result.opex_cooling = result.opex_energy * tco_cfg.cooling_ratio_opex

    # Maintenance (per server per year, scaled by quarter)
    maint_per_server = tco_cfg.maintenance_cost_per_server_per_year * batch.maintenance_multiplier
    result.opex_maintenance = maint_per_server * n / qpy

    # Network OpEx
    if net_cost is not None:
        result.opex_network = net_cost.opex_per_server_per_year * n / qpy

    return result


def compute_provisioning_capex(
    total_power_kw: float,
    cfg: Config,
) -> TcoBreakdown:
    """Compute power and cooling provisioning CapEx for a quarter.

    One-time per-watt costs amortized over the building lifetime,
    then divided into quarterly installments.
    """
    tco_cfg = cfg.tco
    qpy = cfg.simulation.quarters_per_year
    total_power_w = total_power_kw * 1000.0

    result = TcoBreakdown()
    result.capex_power_provisioning = (
        total_power_w * tco_cfg.power_provisioning_cost_per_watt
        / tco_cfg.building_lifetime_years / qpy
    )
    result.capex_cooling_provisioning = (
        total_power_w * tco_cfg.cooling_provisioning_cost_per_watt
        / tco_cfg.building_lifetime_years / qpy
    )

    # Facility cost: proportional to power usage relative to facility capacity,
    # amortized over building lifetime
    num_facilities = total_power_w / tco_cfg.facility_power_capacity_w
    result.capex_facility = (
        num_facilities * tco_cfg.facility_cost
        / tco_cfg.building_lifetime_years / qpy
    )
    return result


def compute_quarter_tco(
    batches: List[ServerBatch],
    q: int,
    cfg: Config,
    net_cost: Optional[NetworkCost] = None,
    stranded_power_kw: float = 0.0,
) -> TcoBreakdown:
    """Compute total TCO for all batches at quarter *q*."""
    total = TcoBreakdown()

    for batch in batches:
        total += compute_batch_capex(batch, q, cfg, net_cost)
        total += compute_batch_opex(batch, q, cfg, net_cost)

    # Power provisioning based on total active power + stranded
    total_power_w = sum(
        b.server.tdp * b.num_servers
        for b in batches if b.active_at(q)
    )
    total_power_kw = total_power_w / 1000.0 + stranded_power_kw
    total += compute_provisioning_capex(total_power_kw, cfg)

    return total
