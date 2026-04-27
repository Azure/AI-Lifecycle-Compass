"""Core lifecycle simulation engine.

A single parameterized loop replaces all 8+ copy-pasted simulation functions
from the original notebooks.  Policy-specific behavior is delegated to
:mod:`dc_tco.policies`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .config import Config, PerformanceModelType
from .demand import compute_demand
from .hardware import ServerSpec, build_gpu_roadmap, filter_roadmap, gpu_to_server
from .models import get_model_at_quarter, project_model_sizes
from .networking import NetworkCost, get_network_cost
from .performance import predict_throughput, predict_throughput_simplified
from .policies import (
    DisaggregatedPolicy,
    get_policy,
)
from .tco import (
    ServerBatch,
    TcoBreakdown,
    compute_quarter_tco,
    compute_racks,
)


def quarter_label(q: int, base_year: int, quarters_per_year: int = 4) -> str:
    """Convert a quarter index to a human-readable label like '2029 Q3'.

    Parameters
    ----------
    q : int
        Zero-based quarter index.
    base_year : int
        Calendar year of simulation start (e.g. 2016).
    quarters_per_year : int
        Quarters per year (default 4).
    """
    year = base_year + q // quarters_per_year
    qn = (q % quarters_per_year) + 1
    return f"{year} Q{qn}"


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class QuarterState:
    """Snapshot of the datacenter state at a single quarter."""
    quarter: int
    year: int
    demand: float                       # req/s
    model_size_B: float                 # Billions of parameters
    total_servers: int
    total_racks: int
    total_capacity: float               # tokens/s (goodput)
    shortfall: float                    # unmet demand (tokens/s)
    servers_added: int
    servers_decommissioned: int
    stranded_power_kw: float
    tco: TcoBreakdown
    gen_server_counts: Dict[str, int]   # code_name -> num_servers


@dataclass
class SimulationResult:
    """Complete simulation output."""
    quarterly_states: List[QuarterState]
    all_batches: List[ServerBatch]
    total_tco: float
    tco_by_year: Dict[int, TcoBreakdown]
    tco_breakdown: TcoBreakdown         # Cumulative breakdown
    config: Config

    @property
    def total_years(self) -> int:
        return self.config.simulation.total_years

    @property
    def num_quarters(self) -> int:
        return self.config.simulation.num_quarters


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(cfg: Config) -> SimulationResult:
    """Run the full datacenter lifecycle simulation.

    Parameters
    ----------
    cfg : Config
        Fully populated configuration.

    Returns
    -------
    SimulationResult
        Complete simulation output including quarterly states, batches,
        and TCO breakdown.
    """
    qpy = cfg.simulation.quarters_per_year
    num_quarters = cfg.simulation.num_quarters
    tco_cfg = cfg.tco

    # --- Build roadmap and model projections ---
    # Build GPU roadmap, apply scenario shocks, then convert to servers
    from .scenarios import apply_hw_capability_shock, apply_price_shock
    gpu_roadmap = build_gpu_roadmap(cfg)
    gpu_roadmap = apply_hw_capability_shock(gpu_roadmap, cfg)
    gpu_roadmap = apply_price_shock(gpu_roadmap, cfg)
    full_roadmap = [gpu_to_server(g, cfg) for g in gpu_roadmap]
    roadmap = filter_roadmap(full_roadmap, cfg.policy.skip_generations)
    model_sizes = project_model_sizes(cfg)
    net_cost = get_network_cost(cfg)
    policy = get_policy(cfg)
    is_disaggregated = isinstance(policy, DisaggregatedPolicy)

    # --- State ---
    batches: List[ServerBatch] = []
    next_batch_id = 0
    quarterly_states: List[QuarterState] = []
    tco_by_year: Dict[int, TcoBreakdown] = defaultdict(TcoBreakdown)

    # Track which GPU generation was most recently released
    prev_available_count = 0

    for q in range(num_quarters):
        year = q // qpy

        # ------------------------------------------------------------------
        # 1. Demand and model size
        # ------------------------------------------------------------------
        demand = compute_demand(q, cfg)
        model_info = get_model_at_quarter(q, model_sizes, cfg)

        # Effective model sizes for capacity planning
        migration_frac = model_info.migration_fraction
        primary_size_B = model_info.primary_model_size / 1e9
        distilled_size_B = model_info.distilled_model_size / 1e9

        # ------------------------------------------------------------------
        # 2. Available GPU generations
        # ------------------------------------------------------------------
        available_gens = [s for s in roadmap if s.release_quarter <= q]
        new_gen_released = len(available_gens) > prev_available_count
        prev_available_count = len(available_gens)

        # ------------------------------------------------------------------
        # 3. Policy decision
        # ------------------------------------------------------------------
        active_batches = [b for b in batches if b.active_at(q)]
        decision = policy.on_quarter(q, active_batches, available_gens, new_gen_released, cfg)

        # Apply forced decommissions
        servers_decom = 0
        for bid in decision.force_decommission:
            for b in batches:
                if b.batch_id == bid and b.active_at(q):
                    servers_decom += b.num_servers
                    b.decom_q = q

        # Apply decom overrides (e.g., extend lifetime)
        for bid, new_decom in decision.decom_overrides.items():
            for b in batches:
                if b.batch_id == bid:
                    b.decom_q = new_decom

        # Apply maintenance overrides
        for bid, mult in decision.maintenance_overrides.items():
            for b in batches:
                if b.batch_id == bid:
                    b.maintenance_multiplier = mult

        # ------------------------------------------------------------------
        # 4. Natural decommissions
        # ------------------------------------------------------------------
        for b in batches:
            if b.active_at(q) and q >= b.decom_q:
                servers_decom += b.num_servers

        # Refresh active list after decommissions
        active_batches = [b for b in batches if b.active_at(q)]

        # ------------------------------------------------------------------
        # 5. Compute current capacity
        # ------------------------------------------------------------------
        if is_disaggregated:
            total_capacity = _compute_disaggregated_capacity(
                active_batches, primary_size_B, distilled_size_B,
                migration_frac, cfg, net_cost,
            )
        else:
            total_capacity = _compute_capacity(
                active_batches, primary_size_B, distilled_size_B,
                migration_frac, cfg, net_cost,
            )
        # ------------------------------------------------------------------
        # 6. Provision new servers if shortfall
        # ------------------------------------------------------------------
        servers_added = 0
        shortfall = demand - total_capacity

        if shortfall > 0 and available_gens:
            gen_idx = decision.buy_gen_index
            if gen_idx is None:
                gen_idx = len(available_gens) - 1
            gen_idx = min(gen_idx, len(available_gens) - 1)
            new_server = available_gens[gen_idx]

            # Compute goodput of new server for current model mix
            # Match the weighted throughput used in _compute_capacity
            gp_primary = _predict(
                primary_size_B, new_server, cfg,
                net_cost.performance_multiplier,
            )
            gp_distilled = _predict(
                distilled_size_B, new_server, cfg,
                net_cost.performance_multiplier,
            )
            goodput = migration_frac * gp_primary + (1 - migration_frac) * gp_distilled

            if goodput > 0:
                num_new = int(math.ceil(shortfall / goodput))
                n_racks = compute_racks(num_new, new_server.tdp, tco_cfg.rack_power_w)
                lifetime_q = tco_cfg.server_lifetime_years * qpy

                batch = ServerBatch(
                    batch_id=next_batch_id,
                    server=new_server,
                    start_q=q,
                    decom_q=q + lifetime_q,
                    num_servers=num_new,
                    num_racks=n_racks,
                )
                batches.append(batch)
                next_batch_id += 1
                servers_added = num_new

                # Recompute capacity after provision
                active_batches = [b for b in batches if b.active_at(q)]
                if is_disaggregated:
                    total_capacity = _compute_disaggregated_capacity(
                        active_batches, primary_size_B, distilled_size_B,
                        migration_frac, cfg, net_cost,
                    )
                else:
                    total_capacity = _compute_capacity(
                        active_batches, primary_size_B, distilled_size_B,
                        migration_frac, cfg, net_cost,
                    )
                shortfall = max(demand - total_capacity, 0)

        # ------------------------------------------------------------------
        # 7. Stranded power
        # ------------------------------------------------------------------
        stranded_kw = _compute_stranded_power(active_batches, tco_cfg.rack_power_w)

        # ------------------------------------------------------------------
        # 8. TCO for this quarter
        # ------------------------------------------------------------------
        quarter_tco = compute_quarter_tco(
            batches, q, cfg, net_cost, stranded_kw
        )
        tco_by_year[year] += quarter_tco

        # ------------------------------------------------------------------
        # 9. Record state
        # ------------------------------------------------------------------
        gen_counts: Dict[str, int] = defaultdict(int)
        total_servers = 0
        total_racks = 0
        for b in active_batches:
            gen_counts[b.server.code_name] += b.num_servers
            total_servers += b.num_servers
            total_racks += b.num_racks

        quarterly_states.append(QuarterState(
            quarter=q,
            year=year,
            demand=demand,
            model_size_B=primary_size_B,
            total_servers=total_servers,
            total_racks=total_racks,
            total_capacity=total_capacity,
            shortfall=shortfall,
            servers_added=servers_added,
            servers_decommissioned=servers_decom,
            stranded_power_kw=stranded_kw,
            tco=quarter_tco,
            gen_server_counts=dict(gen_counts),
        ))

    # --- Aggregate results ---
    cumulative_tco = TcoBreakdown()
    for t in tco_by_year.values():
        cumulative_tco += t

    return SimulationResult(
        quarterly_states=quarterly_states,
        all_batches=batches,
        total_tco=cumulative_tco.total,
        tco_by_year=dict(tco_by_year),
        tco_breakdown=cumulative_tco,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Capacity computation helpers
# ---------------------------------------------------------------------------

def _predict(
    model_size_B: float,
    server: ServerSpec,
    cfg: Config,
    network_multiplier: float = 1.0,
) -> float:
    """Route throughput prediction through the configured model.

    Uses predict_throughput() (which dispatches to theoretical, simplified,
    or CSV) when model_type != 'simplified'; otherwise calls the simplified
    model directly with oversubscription and network multiplier.
    """
    if cfg.performance.model_type == PerformanceModelType.SIMPLIFIED:
        return predict_throughput_simplified(
            model_size_B, server,
            cfg.policy.oversubscription_factor,
            network_multiplier,
        )
    return predict_throughput(model_size_B, server, cfg, network_multiplier)


def _compute_capacity(
    active_batches: List[ServerBatch],
    primary_size_B: float,
    distilled_size_B: float,
    migration_frac: float,
    cfg: Config,
    net_cost: NetworkCost,
) -> float:
    """Compute total fleet capacity accounting for model migration."""
    total = 0.0
    for b in active_batches:
        gp_primary = _predict(
            primary_size_B, b.server, cfg,
            net_cost.performance_multiplier,
        )
        gp_distilled = _predict(
            distilled_size_B, b.server, cfg,
            net_cost.performance_multiplier,
        )
        # If in migration: frac on full model, (1-frac) on distilled
        effective_gp = migration_frac * gp_primary + (1 - migration_frac) * gp_distilled
        total += effective_gp * b.num_servers

    return total


def _compute_disaggregated_capacity(
    active_batches: List[ServerBatch],
    primary_size_B: float,
    distilled_size_B: float,
    migration_frac: float,
    cfg: Config,
    net_cost: NetworkCost,
) -> float:
    """Compute capacity for disaggregated mode (prefill on newer, decode on older).

    Newer servers handle prefill (compute-bound), older servers handle
    decode (memory-bound).
    """
    prefill_frac = cfg.workload.prefill_fraction

    # Sort batches: newest first for prefill, oldest first for decode
    sorted_by_release = sorted(
        active_batches,
        key=lambda b: b.server.release_quarter,
        reverse=True,
    )

    # Simple split: first half (newer) does prefill, second half (older) does decode
    n = len(sorted_by_release)
    if n == 0:
        return 0.0

    total = 0.0
    sizing_B = max(primary_size_B, distilled_size_B)

    for i, b in enumerate(sorted_by_release):
        prefill_perf = _predict(
            sizing_B, b.server, cfg,
            net_cost.performance_multiplier,
        )
        decode_perf = _predict(
            sizing_B, b.server, cfg,
            net_cost.performance_multiplier,
        )

        # Weighted by prefill/decode fraction
        if i < n // 2:
            # Newer servers prioritized for prefill
            total += prefill_perf * b.num_servers * prefill_frac
        else:
            # Older servers for decode
            total += decode_perf * b.num_servers * (1 - prefill_frac)

    return total


def _compute_stranded_power(
    active_batches: List[ServerBatch],
    rack_power_w: float,
) -> float:
    """Compute stranded (wasted) power from partially filled racks.

    Returns stranded power in kW.
    """
    stranded_w = 0.0
    for b in active_batches:
        servers_per_rack = max(1, int(math.floor(rack_power_w / b.server.tdp)))
        remaining = b.num_servers
        for _ in range(b.num_racks):
            used = min(servers_per_rack, remaining)
            remaining -= used
            rack_used_power = used * b.server.tdp
            stranded_w += max(rack_power_w - rack_used_power, 0)

    return stranded_w / 1000.0
