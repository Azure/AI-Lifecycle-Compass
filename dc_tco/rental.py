"""Rental (cloud) TCO calculator — SemiAnalysis-style 8-component model.

Computes monthly and contract-term TCO for GPU cluster rentals, including
direct costs (GPU, storage, networking, control plane, support) and indirect
costs (goodput loss, setup, debugging).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from .config import Config, GoodputMode


HOURS_PER_MONTH = 30 * 24  # 720 hours in a month


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class RentalTcoBreakdown:
    """Monthly rental TCO broken down by the 8 SemiAnalysis components."""
    gpu: float = 0.0
    storage: float = 0.0
    networking: float = 0.0
    control_plane: float = 0.0
    support: float = 0.0
    goodput: float = 0.0
    setup: float = 0.0
    debugging: float = 0.0

    @property
    def direct_cost(self) -> float:
        """Sum of line-item costs (GPU + storage + networking + control plane)."""
        return self.gpu + self.storage + self.networking + self.control_plane

    @property
    def total_before_goodput(self) -> float:
        """Total excluding goodput (what shows on invoice)."""
        return (self.direct_cost + self.support +
                self.setup + self.debugging)

    @property
    def total(self) -> float:
        return self.total_before_goodput + self.goodput

    def __iadd__(self, other: RentalTcoBreakdown) -> RentalTcoBreakdown:
        self.gpu += other.gpu
        self.storage += other.storage
        self.networking += other.networking
        self.control_plane += other.control_plane
        self.support += other.support
        self.goodput += other.goodput
        self.setup += other.setup
        self.debugging += other.debugging
        return self

    def __add__(self, other: RentalTcoBreakdown) -> RentalTcoBreakdown:
        result = RentalTcoBreakdown()
        result += self
        result += other
        return result


@dataclass
class GoodputMetrics:
    """Goodput analysis — both cost and utilization impact."""
    goodput_expense: float = 0.0       # $/month wasted on failures
    effective_gpu_hours: float = 0.0   # Useful GPU-hours per month
    nominal_gpu_hours: float = 0.0     # Total GPU-hours paid for
    utilization: float = 1.0           # effective / nominal (0–1)


@dataclass
class RentalTcoResult:
    """Complete rental TCO output."""
    monthly: RentalTcoBreakdown
    monthly_series: List[RentalTcoBreakdown]
    contract_total: RentalTcoBreakdown
    goodput_metrics: GoodputMetrics
    cost_per_gpu_hour_effective: float  # $/GPU-hr accounting for all costs
    config: Config


# ---------------------------------------------------------------------------
# Goodput formulas (SemiAnalysis "Grand Unifying Theory of Goodput")
# ---------------------------------------------------------------------------

def goodput_checkpoint_cold(cfg: Config) -> GoodputMetrics:
    """Goodput expense: jobs wait for cold repair before restart.

    G_chkpt-cold = {[max(t_id, t_chkpt/2) + t_init] * j_size
                    + t_repair * b_radius} * #failures * $/GPU-hr
    """
    g = cfg.rental.goodput
    r = cfg.rental

    t_id_hr = g.time_to_identify_min / 60.0
    t_chkpt_half_hr = (g.checkpoint_interval_min / 2.0) / 60.0
    t_init_hr = g.job_init_time_min / 60.0
    t_repair_hr = g.repair_time_min / 60.0

    wasted_per_failure = (
        (max(t_id_hr, t_chkpt_half_hr) + t_init_hr) * g.avg_job_size_gpus
        + t_repair_hr * g.blast_radius_gpus
    )

    gpu_price = _effective_gpu_price(r)
    expense = wasted_per_failure * g.num_failures_per_month * gpu_price
    return _build_goodput_metrics(expense, r)


def goodput_checkpoint_hot(cfg: Config) -> GoodputMetrics:
    """Goodput expense: jobs restart immediately on hot spare.

    G_chkpt-hot = [max(t_id, t_chkpt/2) + t_init + t_repair]
                  * j_size * #failures * $/GPU-hr
    """
    g = cfg.rental.goodput
    r = cfg.rental

    t_id_hr = g.time_to_identify_min / 60.0
    t_chkpt_half_hr = (g.checkpoint_interval_min / 2.0) / 60.0
    t_init_hr = g.job_init_time_min / 60.0
    t_repair_hr = g.repair_time_min / 60.0

    wasted_per_failure = (
        (max(t_id_hr, t_chkpt_half_hr) + t_init_hr + t_repair_hr)
        * g.avg_job_size_gpus
    )

    gpu_price = _effective_gpu_price(r)
    expense = wasted_per_failure * g.num_failures_per_month * gpu_price
    return _build_goodput_metrics(expense, r)


def goodput_fault_tolerant(cfg: Config) -> GoodputMetrics:
    """Goodput expense: fault-tolerant framework (TorchFT, TorchPass, etc.).

    G_tolerant = [(t_id + t_failover) * j_size + t_repair * b_radius]
                 * #failures * $/GPU-hr
    """
    g = cfg.rental.goodput
    r = cfg.rental

    t_id_hr = g.time_to_identify_min / 60.0
    t_failover_hr = g.failover_time_min / 60.0
    t_repair_hr = g.repair_time_min / 60.0

    wasted_per_failure = (
        (t_id_hr + t_failover_hr) * g.avg_job_size_gpus
        + t_repair_hr * g.blast_radius_gpus
    )

    gpu_price = _effective_gpu_price(r)
    expense = wasted_per_failure * g.num_failures_per_month * gpu_price

    # Additional overhead from fault-tolerant frameworks
    nominal_hrs = r.num_gpus * HOURS_PER_MONTH
    overhead_frac = max(g.network_overhead_pct, g.memory_overhead_pct)
    overhead_expense = nominal_hrs * gpu_price * overhead_frac

    return _build_goodput_metrics(expense + overhead_expense, r)


# ---------------------------------------------------------------------------
# Monthly TCO computation
# ---------------------------------------------------------------------------

def compute_rental_monthly(cfg: Config) -> RentalTcoBreakdown:
    """Compute a single month's rental TCO breakdown."""
    r = cfg.rental
    gpu_price = _effective_gpu_price(r)

    result = RentalTcoBreakdown()

    # 1. GPU cost
    on_demand_gpus = r.num_gpus * (1 - r.spot_fraction)
    spot_gpus = r.num_gpus * r.spot_fraction
    base_gpu = (
        on_demand_gpus * gpu_price * HOURS_PER_MONTH
        + spot_gpus * gpu_price * (1 - r.spot_discount_pct) * HOURS_PER_MONTH
    )
    result.gpu = base_gpu * (1 + r.orchestration_premium_pct)

    # 2. Storage
    total_tb = r.storage.tb_per_gpu * r.num_gpus
    total_gb = total_tb * 1024.0
    cold_frac = max(0.0, min(1.0, 1.0 - r.storage.hot_fraction - r.storage.warm_fraction))
    result.storage = total_gb * (
        r.storage.hot_fraction * r.storage.hot_cost_per_gb_mo
        + r.storage.warm_fraction * r.storage.warm_cost_per_gb_mo
        + cold_frac * r.storage.cold_cost_per_gb_mo
    )

    # 3. Networking
    result.networking = (
        r.networking.cost_per_month
        + r.networking.egress_per_gb * r.networking.egress_gb_per_month
    )

    # 4. Control plane
    result.control_plane = (
        r.control_plane.cost_per_hour
        * r.control_plane.num_nodes
        * HOURS_PER_MONTH
    )

    # 5. Support (percentage of direct costs)
    result.support = result.direct_cost * r.support.uplift_pct

    # 6. Goodput
    goodput_metrics = compute_goodput(cfg)
    result.goodput = goodput_metrics.goodput_expense

    # 7. Setup (amortized over contract term)
    setup_total = (
        r.setup.engineering_hours * r.setup.engineering_rate
        + r.setup.cluster_hours * gpu_price * r.num_gpus
    )
    result.setup = setup_total / max(r.contract_months, 1)

    # 8. Debugging
    result.debugging = (
        r.debugging.engineering_hours_per_month * r.debugging.engineering_rate
        + r.debugging.cluster_hours_per_month * gpu_price * r.num_gpus
    )

    return result


def compute_goodput(cfg: Config) -> GoodputMetrics:
    """Dispatch to the appropriate goodput formula based on config."""
    mode = cfg.rental.goodput.mode
    if mode == GoodputMode.CHECKPOINT_COLD:
        return goodput_checkpoint_cold(cfg)
    elif mode == GoodputMode.CHECKPOINT_HOT:
        return goodput_checkpoint_hot(cfg)
    elif mode == GoodputMode.FAULT_TOLERANT:
        return goodput_fault_tolerant(cfg)
    else:
        raise ValueError(f"Unknown goodput mode: {mode}")


def compute_rental_tco(cfg: Config) -> RentalTcoResult:
    """Compute full rental TCO over the contract term.

    Builds a per-month series, then aggregates. Setup cost is
    amortized; all other costs are recurring monthly.
    """
    months = max(cfg.rental.contract_months, 1)
    monthly = compute_rental_monthly(cfg)

    # Build monthly series (currently uniform; extensible for
    # price changes, renewals, spot fluctuations)
    series: List[RentalTcoBreakdown] = []
    contract_total = RentalTcoBreakdown()
    for _ in range(months):
        series.append(replace(monthly))
        contract_total += monthly

    # Effective cost metrics
    goodput_metrics = compute_goodput(cfg)
    effective_cost_per_gpu_hr = (
        contract_total.total / max(goodput_metrics.effective_gpu_hours
                                   * months, 1.0)
    )

    return RentalTcoResult(
        monthly=monthly,
        monthly_series=series,
        contract_total=contract_total,
        goodput_metrics=goodput_metrics,
        cost_per_gpu_hour_effective=effective_cost_per_gpu_hr,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_gpu_price(r) -> float:
    """GPU price after discount."""
    return r.gpu_cost_per_hour * (1 - r.discount_pct)


def _build_goodput_metrics(expense: float, r) -> GoodputMetrics:
    """Build GoodputMetrics from expense and rental config."""
    nominal_hrs = r.num_gpus * HOURS_PER_MONTH
    gpu_price = _effective_gpu_price(r)
    nominal_cost = nominal_hrs * gpu_price

    utilization = 1.0
    if nominal_cost > 0:
        utilization = max(0.0, 1.0 - expense / nominal_cost)

    effective_hrs = nominal_hrs * utilization
    return GoodputMetrics(
        goodput_expense=expense,
        effective_gpu_hours=effective_hrs,
        nominal_gpu_hours=nominal_hrs,
        utilization=utilization,
    )
